using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace ViewModel
{
    class AsyncRelayCommand : ICommand
    {
        private readonly Func<object, Task> execute;
        private readonly Func<object, bool> canExecute;
        private bool is_busy { get; set; }
        public AsyncRelayCommand(Func<object, Task> execute, Func<object, bool> canExecute = null)
        {
            this.execute = execute;
            this.canExecute = canExecute;
        }

        public event EventHandler CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }

        public bool CanExecute(object parameter)
        {
            if (is_busy)
                return false;
            else
                return canExecute == null ? true : canExecute(parameter);
        }

        public void Execute(object parameter)
        {
            if (!is_busy)
            {
                is_busy = true;
                execute?.Invoke(parameter).ContinueWith(_ =>
                {
                    is_busy = false;
                    CommandManager.InvalidateRequerySuggested();
                }, CancellationToken.None, TaskContinuationOptions.None, TaskScheduler.FromCurrentSynchronizationContext());
            }
        }

    }
}
