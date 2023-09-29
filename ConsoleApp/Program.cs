using System;
using SixLabors.ImageSharp; // Из одноимённого пакета NuGet
using SixLabors.ImageSharp.PixelFormats;
using System.Linq;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using SixLabors.Fonts;
using ClassLibrary;
using System.Threading.Tasks;
using System.Xml.Linq;
using System.Threading;
using CsvHelper;
using static ClassLibrary.ImageAnalyzer;
using System.Globalization;
using System.IO;

namespace YOLO_csharp
{
    class Program
    {
        public class ObjectEntity
        {
            public string FileName { get; private set; }
            public string ClassName { get; private set; }
            public int X { get; private set; }
            public int Y { get; private set; }
            public int W { get; private set; }
            public int H { get; private set; }
            public ObjectEntity(string fileName, string className, int x, int y, int w, int h)
            {
                this.FileName = fileName;
                this.ClassName = className;
                this.X = x;
                this.Y = y;
                this.W = w;
                this.H = h;
            }
        }
        public static void make_CSV_file(List<Tuple<Image<Rgb24>, ObjectBox>> cut_pairs)
        {
            string[] labels = new string[]
            {
                "aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            };

            List<string> filenames = Enumerable
            .Range(0, cut_pairs.Count)
            .Select(i => $"file{i}.jpg")
            .ToList();

            string pathCsvFile = "fileCSV.csv";

            List<ObjectEntity> objectsCSV = new List<ObjectEntity>();
            for (int i = 0; i < cut_pairs.Count; i++)
            {
                var filename = filenames[i];

                var image = cut_pairs[i].Item1;
                image.SaveAsJpeg(filename);

                var obj = cut_pairs[i].Item2;

                objectsCSV.Add(new ObjectEntity(filename, labels[obj.Class], (int)obj.XMin, (int)obj.YMin, (int)(obj.XMax - obj.XMin), (int)(obj.YMax - obj.YMin)));
            }

            using (StreamWriter streamReader = new StreamWriter(pathCsvFile))
            {
                using (CsvWriter csvReader = new CsvWriter(streamReader, CultureInfo.InvariantCulture))
                {
                    csvReader.WriteRecords(objectsCSV);
                }
            }
        }
        static CancellationTokenSource ctf = new CancellationTokenSource();

        protected static void myHandler(object sender, ConsoleCancelEventArgs args)
        {
            args.Cancel = true;
            ctf.Cancel();
        }
        static async Task Main(string[] args)
        {
            Console.CancelKeyPress += new ConsoleCancelEventHandler(myHandler);

            try
            {
                var image = SixLabors.ImageSharp.Image.Load<Rgb24>(args.FirstOrDefault());
                ImageAnalyzer obj = await ImageAnalyzer.Create(ctf.Token);
                var objects = await obj.Detect(image, ctf.Token);
                var cut_pairs = await obj.CutImage(image, objects, ctf.Token);
                make_CSV_file(cut_pairs);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

        }
    }

}
