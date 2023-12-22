using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc.Testing;
using Newtonsoft.Json;
using Xunit;
using Web;
using System.Text;
using System.Net.Http.Json;

namespace WebAnsweringTests
{
    public class Tests : IClassFixture<WebApplicationFactory<Program>>
    {
        private readonly WebApplicationFactory<Program> webApplicationFactory;
        public Tests(WebApplicationFactory<Program> webApplicationFactory)
        {
            this.webApplicationFactory = webApplicationFactory;
        }
        [Fact]
        public async Task Test1()
        {
            var client = webApplicationFactory.CreateClient();
            string base64 = Convert.ToBase64String(File.ReadAllBytes("C:\\prak7\\task4\\chair.jpg"));
            var response = await client.PostAsJsonAsync("http://localhost:5276/api/Image/post1", new ClientRequest { Base64 = base64 });
            response.EnsureSuccessStatusCode();
            var objects = JsonConvert.DeserializeObject<List<DetectedObject>>(await response.Content.ReadAsStringAsync());
            Assert.NotNull(objects);
            Assert.Single(objects);
            Assert.Equal("chair", objects[0].Class);

        }


    }
}