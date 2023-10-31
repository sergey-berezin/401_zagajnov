using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using ViewModel;
namespace View
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window, IUIServices
    {
        public MainWindow()
        {
            
            InitializeComponent();
            DataContext = new MainViewModel(this);
        }
        //public void ReportError(string message)
        //{
        //    errorBar.Text = message;
        //    stack.Visibility = Visibility.Visible;
        //}
        public string OpenSaveDialog()
        {
            return null;
        }
        public string[] OpenLoadDialog()
        {
            OpenFileDialog dlg = new OpenFileDialog();
            dlg.Multiselect = true;
            if (dlg.ShowDialog() == true)
            {
                return dlg.FileNames;
            }
            return null;
        }

        //private void Button_Click(object sender, RoutedEventArgs e)
        //{
        //    stack.Visibility = Visibility.Collapsed;
        //}
    }
}
