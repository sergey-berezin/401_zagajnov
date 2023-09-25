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

namespace YOLO_csharp
{
    class Program
    {
        static async Task Main(string[] args)
        {
            

            IProgress<int> p = new Progress<int>(progress =>
            {
                Console.WriteLine("Running Step: {0}", progress);
            });

            ImageAnalyzer obj = await ImageAnalyzer.Create(p);
            obj.ImagePreprocessing(args.FirstOrDefault() ?? "chair.jpg");
            await obj.Detect();
        }
    }

}
