using Microsoft.EntityFrameworkCore;

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
}