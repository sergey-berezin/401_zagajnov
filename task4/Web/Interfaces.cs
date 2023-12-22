using ClassLibrary;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using YamlDotNet.Core.Tokens;

namespace Web
{

    public class ClientRequest
    {
        public string Base64 { get; set; }
     
    }
    public class DetectedObject
    {
        public double XMin { get; set; }
        public double YMin { get; set; }
        public double XMax { get; set; }
        public double YMax { get; set; }
        public double Confidence { get; set; }
        public string Class { get; set; }
        public DetectedObject(double xMin, double yMin, double xMax, double yMax, double confidence, string @class)
        {
            XMin = xMin;
            YMin = yMin;
            XMax = xMax;
            YMax = yMax;
            Confidence = confidence;
            Class = @class;
        }
    }
    public interface IImageProcessor
    {
        public Task<(Image<Rgb24>, List<ImageAnalyzer.ObjectBox>)> DoAction(Image<Rgb24> image, CancellationToken token);
    }
    public class Processor : IImageProcessor
    {
        public async Task<(Image<Rgb24>, List<ImageAnalyzer.ObjectBox>)> DoAction(Image<Rgb24> image, CancellationToken token)
        { 
       
            var imageAnalyzer = await ImageAnalyzer.Create(token);
            return await imageAnalyzer.Detect(image, token);
        }
    }




}



