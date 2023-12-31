﻿// <auto-generated />
using System;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;
using Model.DataBase;

#nullable disable

namespace Model.Migrations
{
    [DbContext(typeof(LibraryContext))]
    partial class LibraryContextModelSnapshot : ModelSnapshot
    {
        protected override void BuildModel(ModelBuilder modelBuilder)
        {
#pragma warning disable 612, 618
            modelBuilder
                .HasAnnotation("ProductVersion", "7.0.13")
                .HasAnnotation("Proxies:ChangeTracking", false)
                .HasAnnotation("Proxies:CheckEquality", false)
                .HasAnnotation("Proxies:LazyLoading", true);

            modelBuilder.Entity("Model.DataBase.Image", b =>
                {
                    b.Property<int>("Id")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("INTEGER");

                    b.Property<byte[]>("Bytes")
                        .IsRequired()
                        .HasColumnType("BLOB");

                    b.Property<int>("Height")
                        .HasColumnType("INTEGER");

                    b.Property<string>("Path")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<int>("Width")
                        .HasColumnType("INTEGER");

                    b.HasKey("Id");

                    b.ToTable("Images");
                });

            modelBuilder.Entity("Model.DataBase.Object", b =>
                {
                    b.Property<int>("Id")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("INTEGER");

                    b.Property<int>("Class")
                        .HasColumnType("INTEGER");

                    b.Property<double>("Confidence")
                        .HasColumnType("REAL");

                    b.Property<int?>("ImageId")
                        .HasColumnType("INTEGER");

                    b.Property<double>("XMax")
                        .HasColumnType("REAL");

                    b.Property<double>("XMin")
                        .HasColumnType("REAL");

                    b.Property<double>("YMax")
                        .HasColumnType("REAL");

                    b.Property<double>("YMin")
                        .HasColumnType("REAL");

                    b.HasKey("Id");

                    b.HasIndex("ImageId");

                    b.ToTable("Objects");
                });

            modelBuilder.Entity("Model.DataBase.Object", b =>
                {
                    b.HasOne("Model.DataBase.Image", null)
                        .WithMany("Objects")
                        .HasForeignKey("ImageId");
                });

            modelBuilder.Entity("Model.DataBase.Image", b =>
                {
                    b.Navigation("Objects");
                });
#pragma warning restore 612, 618
        }
    }
}
