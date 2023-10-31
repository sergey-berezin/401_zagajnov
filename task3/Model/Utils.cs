using System;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Linq;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using SixLabors.Fonts;
using System.Threading.Tasks;
using System.Threading;
using static ClassLibrary.ImageAnalyzer;
using System.IO;
using System.Windows.Input;
using System.Collections.ObjectModel;
using ClassLibrary;
using Microsoft.EntityFrameworkCore;
using Model.DataBase;
using System.Runtime.CompilerServices;


namespace Model
{   
    namespace DataBase
    {
        public class Image
        {
            public int Id { get; set; }
            public string Path { get; set; }
            public byte[] Bytes { get; set; }
            public int Height { get; set; }
            public int Width { get; set; }

            virtual public ICollection<Object> Objects { get; set; }
        }

        public class Object
        {
            public int Id { get; set; }
            public double XMin { get; set; }
            public double YMin { get; set; }
            public double XMax { get; set; }
            public double YMax { get; set; }
            public double Confidence { get; set; }
            public int Class { get; set; }

        }


        class LibraryContext : DbContext
        {
            public DbSet<Image> Images { get; set; }
            public DbSet<Object> Objects { get; set; }

            protected override void OnConfiguring(DbContextOptionsBuilder o)
            {
                string fullPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "library.db");

                o.UseLazyLoadingProxies().UseSqlite($"Data Source={fullPath}");
            }
        }
    }

    public class Utils
    {
        static public void DeleteData()
        {
            using (var db = new LibraryContext())
            {
                db.Images.RemoveRange(db.Images);
                db.Objects.RemoveRange(db.Objects);
                db.SaveChanges();
            }
        }
        static public List<(Image<Rgb24>, List<ObjectBox>, string)> GetAllData()
        {
            List<(Image<Rgb24>, List<ObjectBox>, string)> values = new();
            using (var db = new LibraryContext())
            {
                foreach (var image in db.Images)
                    values.Add((SixLabors.ImageSharp.Image.LoadPixelData<Rgb24>(image.Bytes, image.Width, image.Height),
                                image.Objects.Select(obj => new ObjectBox(obj.XMin, obj.YMin, obj.XMax, obj.YMax, obj.Confidence, obj.Class)).ToList(),
                                image.Path));
            }
            return values;

        }
        static public async Task<(Image<Rgb24>, List<ObjectBox>, string)> DetectWithCacheAsync(string path, CancellationToken token)
        {
            using (var db = new LibraryContext())
            {
               

                var image = db.Images.FirstOrDefault(x => x.Path == path);
               
                if (image != null)
                {
                    return (SixLabors.ImageSharp.Image.LoadPixelData<Rgb24>(image.Bytes, image.Width, image.Height),
                            image.Objects.Select(obj => new ObjectBox(obj.XMin, obj.YMin, obj.XMax, obj.YMax, obj.Confidence, obj.Class)).ToList(),
                            image.Path);
                }
                else
                {
                    ImageAnalyzer imageAnalyzer = await ImageAnalyzer.Create(token);
                    var rez = await imageAnalyzer.Detect(SixLabors.ImageSharp.Image.Load<Rgb24>(path), token);

                    var img = rez.Item1;
                    var objs = rez.Item2;

                    byte[] pixels = new byte[img.Width * img.Height * Unsafe.SizeOf<Rgb24>()];
                    img.CopyPixelDataTo(pixels);
                    var newImage = new DataBase.Image() { Path = path, Bytes = pixels};
                    newImage.Objects = new List<DataBase.Object>();
                    newImage.Height = img.Height;
                    newImage.Width = img.Width;
                    foreach (ObjectBox obj in objs)
                    {
                        var newObject = new DataBase.Object() { XMin = obj.XMin, YMin = obj.YMin, XMax = obj.XMax, YMax = obj.YMax, Confidence = obj.Confidence, Class = obj.Class };
                        newImage.Objects.Add(newObject);
                        db.Add(newObject);
                    }
                    db.Add(newImage);
                    db.SaveChanges();

                    return (img, objs, path);
                }

            }
        }

        static public ObservableCollection<Image<Rgb24>> GetImages(ObservableCollection<string> file_names)
        {
            ObservableCollection<Image<Rgb24>> list = new();
            foreach (string file_name in file_names)
                list.Add(SixLabors.ImageSharp.Image.Load<Rgb24>(file_name));
            return list;
        }
        const int TargetSize = 416;
        static string[] labels = new string[]
            {
                "aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            };
        static public Image<Rgb24> Annotate(Image<Rgb24> target, IEnumerable<ObjectBox> objects)
        {
            Image<Rgb24> copy = target.Clone(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(TargetSize, TargetSize),
                    Mode = ResizeMode.Pad
                });
            });
            
            foreach (var objbox in objects)
            {
                copy.Mutate(ctx =>
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
            return copy;
        }
    }
}
