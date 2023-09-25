﻿using System;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Linq;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using SixLabors.Fonts;
using System.Net;
using System.Formats.Asn1;
using CsvHelper;
using System.Globalization;
using SixLabors.ImageSharp.ColorSpaces;
using static System.Net.Mime.MediaTypeNames;
using System.Net.Http;
using Polly;

namespace ClassLibrary
{
    public class ImageAnalyzer
    {
        Image<Rgb24> image { get; set; }
        Image<Rgb24> resized { get; set; }
        List<NamedOnnxValue> inputs { get; set; }
        private ImageAnalyzer()
        {

        }
        //private static Task NetworkDownloader(IProgress<int> p, CancellationToken token)
        //{

        //    return Task.Factory.StartNew(() =>
        //    {
        //        if (!System.IO.File.Exists("tinyyolov2-8.onnx"))
        //            using (var client = new WebClient())
        //            {
        //                //Environment.GetFolderPath(Environment.SpecialFolder.System) + 
        //                //token.ThrowIfCancellationRequested();
        //                client.DownloadFile("https://storage.yandexcloud.net/dotnet4/tinyyolov2-8.onnx", "tinyyolov2-8.onnx");

        //            }
        //        token.ThrowIfCancellationRequested();
        //    }, token, TaskCreationOptions.LongRunning, TaskScheduler.Current);

        //}
        private static async Task NetworkDownloader(IProgress<int> p, CancellationToken token)
        {
            if (!System.IO.File.Exists("tinyyolov2-8.onnx"))
            {
                var jitterer = new Random();

                var retryPolicy = Policy
                    .Handle<HttpRequestException>()
                    .WaitAndRetryAsync(5,
                        retryAttempt => TimeSpan.FromSeconds(Math.Pow(2, retryAttempt))  // exponential back-off: 2, 4, 8 etc
                                      + TimeSpan.FromMilliseconds(jitterer.Next(0, 1000)));  // plus some jitter: up to 1 second

                using (var httpClient = new HttpClient())
                {
                    var buffer = await retryPolicy.ExecuteAsync(async () =>
                    {
                        Console.WriteLine("Getting data...");
                        return await httpClient.GetByteArrayAsync("https://storage.yandexcloud.net/dotnet4/tinyyolov2-8.onnx", token);
                    }); 
                    await File.WriteAllBytesAsync("tinyyolov2-8.onnx", buffer, token);
                }
            }
        }

        public static async Task<ImageAnalyzer> Create(IProgress<int> p)
        {
            try
            {
                CancellationTokenSource ctf = new CancellationTokenSource();
                var task = NetworkDownloader(p, ctf.Token);
                //ctf.Cancel();
                await task;
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.Message);
                Environment.Exit(1);
            }
            
            return new ImageAnalyzer();
        }

        string[] labels = new string[]
            {
                "aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            };

        float Sigmoid(float value)
        {
            var e = (float)Math.Exp(value);
            return e / (1.0f + e);
        }

        float[] Softmax(float[] values)
        {
            var exps = values.Select(v => Math.Exp(v));
            var sum = exps.Sum();
            return exps.Select(e => (float)(e / sum)).ToArray();
        }

        public void ImagePreprocessing(string image_path)
        {
            image = SixLabors.ImageSharp.Image.Load<Rgb24>(image_path);

            int imageWidth = image.Width;
            int imageHeight = image.Height;

            // Размер изображения
            const int TargetSize = 416;

            // Изменяем размер изображения до 416 x 416
            resized = image.Clone(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(TargetSize, TargetSize),
                    Mode = ResizeMode.Pad // Дополнить изображение до указанного размера с сохранением пропорций
                });
            });

            // Перевод пикселов в тензор и нормализация
            var input = new DenseTensor<float>(new[] { 1, 3, TargetSize, TargetSize });


            resized.ProcessPixelRows(pa =>
            {
                for (int y = 0; y < TargetSize; y++)
                {
                    Span<Rgb24> pixelSpan = pa.GetRowSpan(y);
                    for (int x = 0; x < TargetSize; x++)
                    {
                        input[0, 0, y, x] = pixelSpan[x].R;
                        input[0, 1, y, x] = pixelSpan[x].G;
                        input[0, 2, y, x] = pixelSpan[x].B;
                    }
                }
            });

            // Подготавливаем входные данные нейросети. Имя input задано в файле модели
            inputs = new List<NamedOnnxValue>
            {
               NamedOnnxValue.CreateFromTensor("image", input),
            };
        }
        private List<ObjectBox> GetObjects()
        {
            int imageWidth = image.Width;
            int imageHeight = image.Height;
            const int TargetSize = 416;
            // Вычисляем предсказание нейросетью
            using var session = new InferenceSession("tinyyolov2-8.onnx");
            //foreach(var n in session.InputNames)
            //    Console.WriteLine(n);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Получаем результаты
            var outputs = results.First().AsTensor<float>();

            //foreach (var d in outputs.Dimensions)
            //    Console.WriteLine(d);


            const int CellCount = 13; // 13x13 ячеек
            const int BoxCount = 5; // 5 прямоугольников в каждой ячейке
            const int ClassCount = 20; // 20 классов



            int IndexOfMax(float[] values)
            {
                int idx = 0;
                for (int i = 1; i < values.Length; i++)
                    if (values[i] > values[idx])
                        idx = i;
                return idx;
            }

            var anchors = new (double, double)[]
            {
               (1.08, 1.19),
               (3.42, 4.41),
               (6.63, 11.38),
               (9.42, 5.11),
               (16.62, 10.52)
            };

            int cellSize = TargetSize / CellCount;

            var boundingBoxes = resized.Clone();

            List<ObjectBox> objects = new();

            for (var row = 0; row < CellCount; row++)
                for (var col = 0; col < CellCount; col++)
                    for (var box = 0; box < BoxCount; box++)
                    {
                        var rawX = outputs[0, (5 + ClassCount) * box, row, col];
                        var rawY = outputs[0, (5 + ClassCount) * box + 1, row, col];

                        var rawW = outputs[0, (5 + ClassCount) * box + 2, row, col];
                        var rawH = outputs[0, (5 + ClassCount) * box + 3, row, col];

                        var x = (float)((col + Sigmoid(rawX)) * cellSize);
                        var y = (float)((row + Sigmoid(rawY)) * cellSize);

                        var w = (float)(Math.Exp(rawW) * anchors[box].Item1 * cellSize);
                        var h = (float)(Math.Exp(rawH) * anchors[box].Item2 * cellSize);

                        var conf = Sigmoid(outputs[0, (5 + ClassCount) * box + 4, row, col]);

                        if (conf > 0.5)
                        {
                            var classes
                            = Enumerable
                            .Range(0, ClassCount)
                            .Select(i => outputs[0, (5 + ClassCount) * box + 5 + i, row, col])
                            .ToArray();
                            objects.Add(new ObjectBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, conf, IndexOfMax(Softmax(classes))));
                        }

                        if (conf > 0.01)
                        {
                            boundingBoxes.Mutate(ctx =>
                            {
                                ctx.DrawPolygon(
                                    Pens.Solid(Color.Green, 1),
                                    new PointF[] {
                                        new PointF(x - w / 2, y - h / 2),
                                        new PointF(x + w / 2, y - h / 2),
                                        new PointF(x + w / 2, y + h / 2),
                                        new PointF(x - w / 2, y + h / 2)
                                    });
                            });
                        }
                    }
            boundingBoxes.Save("boundingboxes.jpg");

            void Annotate(Image<Rgb24> target, IEnumerable<ObjectBox> objects)
            {
                foreach (var objbox in objects)
                {
                    target.Mutate(ctx =>
                    {
                        ctx.DrawPolygon(
                            Pens.Solid(Color.Blue, 2),
                            new PointF[] {
                                new PointF((float)objbox.XMin, (float)objbox.YMin),
                                new PointF((float)objbox.XMin, (float)objbox.YMax),
                                new PointF((float)objbox.XMax, (float)objbox.YMax),
                                new PointF((float)objbox.XMax, (float)objbox.YMin)
                            });

                        ctx.DrawText(
                            $"{labels[objbox.Class]}",
                            SystemFonts.Families.First().CreateFont(16),
                            Color.Blue,
                            new PointF((float)objbox.XMin, (float)objbox.YMax));
                    });
                }
            }

            var annotated = resized.Clone();
            Annotate(annotated, objects);
            annotated.SaveAsJpeg("annotated.jpg");

            // Убираем дубликаты
            for (int i = 0; i < objects.Count; i++)
            {
                var o1 = objects[i];
                for (int j = i + 1; j < objects.Count;)
                {
                    var o2 = objects[j];
                    Console.WriteLine($"IoU({i},{j})={o1.IoU(o2)}");
                    if (o1.Class == o2.Class && o1.IoU(o2) > 0.6)
                    {
                        if (o1.Confidence < o2.Confidence)
                        {
                            objects[i] = o1 = objects[j];
                        }
                        objects.RemoveAt(j);
                    }
                    else
                    {
                        j++;
                    }
                }
            }

            
            var final = resized.Clone();
            Annotate(final, objects);
            final.SaveAsJpeg("final.jpg");
            return objects;
        }
        public record ObjectBox(double XMin, double YMin, double XMax, double YMax, double Confidence, int Class)
        {
            public double IoU(ObjectBox b2) =>
                (Math.Min(XMax, b2.XMax) - Math.Max(XMin, b2.XMin)) * (Math.Min(YMax, b2.YMax) - Math.Max(YMin, b2.YMin)) /
                ((Math.Max(XMax, b2.XMax) - Math.Min(XMin, b2.XMin)) * (Math.Max(YMax, b2.YMax) - Math.Min(YMin, b2.YMin)));
        }
        public async Task Detect()
        {
            var objects = GetObjects();
            List<string> filenames = new List<string>();
            List<Image<Rgb24>> cut_images = new List<Image<Rgb24>>();
            await foreach (var obj in make_files_with_object(objects))
            {
                cut_images.Add(obj.Key);
                filenames.Add(obj.Value);
                obj.Key.SaveAsJpeg(obj.Value);
            }
            make_CSV_file(objects, filenames);
        }
        

        async IAsyncEnumerable<KeyValuePair<Image<Rgb24>, string>> make_files_with_object(List<ObjectBox> objects)
        {
            List<string> filenames = Enumerable
            .Range(0, objects.Count)
            .Select(i => $"file{i}.jpg")
            .ToList();

            for (int i = 0; i < objects.Count; i++)
            {
                ObjectBox obj = objects[i];
                var obj_image = resized.Clone(x =>
                {
                    x.Crop(new Rectangle((int)obj.XMin, (int)obj.YMin, (int)(obj.XMax - obj.XMin), (int)(obj.YMax - obj.YMin)));
                });
                yield return new KeyValuePair<Image<Rgb24>, string> (obj_image, filenames[i]);
                //obj_image.SaveAsJpeg(filenames[i]);
            }
        }
        public class ObjectEntity
        {
            public string fileName { get; set; }
            public string className { get; set; }
            public int x { get; set; }
            public int y { get; set; }
            public int w { get; set; }
            public int h { get; set; }
            public ObjectEntity(string fileName, string className, int x, int y, int w, int h)
            {
                this.fileName = fileName;
                this.className = className;
                this.x = x;
                this.y = y;
                this.w = w;
                this.h = h;
            }
        }
        private void make_CSV_file(List<ObjectBox> objects, List<string> filenames)
        {
            string pathCsvFile = "fileCSV.csv";

            List<ObjectEntity> objectsCSV = new List<ObjectEntity>();
            for(int i = 0; i < objects.Count; i++)
            {
                var obj = objects[i];
                var filename = filenames[i];
                objectsCSV.Add(new ObjectEntity(filename, labels[obj.Class], (int)obj.XMin, (int)obj.YMin, (int)(obj.XMax - obj.XMin), (int)(obj.YMax - obj.YMin)));
            }

            using (StreamWriter streamReader = new StreamWriter(pathCsvFile))
            {
                using (CsvWriter csvReader = new CsvWriter(streamReader, CultureInfo.InvariantCulture))
                {
                    //csvReader.Configuration.Delimiter = ",";
                    csvReader.WriteRecords(objectsCSV);
                }
            }
        }
    }
}
