using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;


namespace Web.Controllers
{

    [ApiController]
    [Route("api/[controller]")]
    public class ImageController : ControllerBase
    {

        private readonly IImageProcessor iP;

        private string[] labels = new string[]
            {
                "aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            };

        public ImageController(IImageProcessor ip)
        {
            this.iP = ip;
        }

        [HttpPost("post1")]
        public async Task<ActionResult<DetectedObject[]>> Post([FromBody] ClientRequest clientRequest)
        {
            byte[] picture = Convert.FromBase64String(clientRequest.Base64);
            Image<Rgb24> image = SixLabors.ImageSharp.Image.Load<Rgb24>(picture);
            var rez = await iP.DoAction(image, CancellationToken.None);
            var objectboxes = rez.Item2;
            List<DetectedObject> objects = objectboxes.Select(obj => new DetectedObject(obj.XMin, obj.YMin, obj.XMax, obj.YMax, obj.Confidence, labels[obj.Class])).ToList();
            return objects.ToArray();
        }

        [HttpPost("post2")]
        public ActionResult<string> Post_ResIm([FromBody] ClientRequest clientRequest)
        {
            byte[] picture = Convert.FromBase64String(clientRequest.Base64);
            Image<Rgb24> image = SixLabors.ImageSharp.Image.Load<Rgb24>(picture);
            const int TargetSize = 416;
            
            var resized = image.Clone(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(TargetSize, TargetSize),
                    Mode = ResizeMode.Pad // Дополнить изображение до указанного размера с сохранением пропорций
                });
            });

            return Content(resized.ToBase64String(PngFormat.Instance));
        }
    }
}