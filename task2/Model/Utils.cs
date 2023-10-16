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

namespace Model
{
    public class Utils
    {
        static public ObservableCollection<Image<Rgb24>> GetImages(ObservableCollection<string> file_names)
        {
            ObservableCollection<Image<Rgb24>> list = new();
            foreach (string file_name in file_names)
                list.Add(Image.Load<Rgb24>(file_name));
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
