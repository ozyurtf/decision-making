import csv
import datetime
from collections import defaultdict
import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

COMMON_TRAIN_FORMAT = [('step', 'S', 'int'),
                       ('actor_loss', 'L', 'float'),
                       ('total_time', 'T', 'time')]

COMMON_EVAL_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                      ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                      ('episode_reward', 'R', 'float'),
                      ('imitation_reward', 'R_i', 'float'),
                      ('total_time', 'T', 'time')]

class AverageMeter(object):
    """
    Keep track of a running average 
    of specific metrics that are tracked/stored
    during model training/evaluaton processes.
    """
    def __init__(self):
        # Initialize a variable to keep track of the sum of 
        # all values added.
        self._sum = 0
        
        # Initialize a variable to keep track of how many values 
        # are added.
        self._count = 0

    def update(self, value, n=1):
        # Add the provided 'value' to the running sum.
        self._sum += value
        
        # Increase the variable that keeps track of 
        # the number of values added by 'n'
        self._count += n

    def value(self):
        # Return the average.
        return self._sum / max(1, self._count)

class MetersGroup(object):
    def __init__(self, csv_file_name, formating):
        self._csv_file_name = csv_file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None

    def log(self, key, value, n=1):
        # Access (or create if it doesn't already exist) 
        # the AverageMeter class for the specified 'key' 
        # with self._meters[key] and 
        # add the 'value' to the running sum and 
        # increment the running count by 'n' which 
        # is the number of samples that are part of for 
        # the calculation of 'value'.
        self._meters[key].update(value, n)

    def _prime_meters(self):
        # Create an empty dictionary to store 
        # the metric names and their current values.
        data = dict()
        
        for key, meter in self._meters.items():
            # If the metric name starts with 'train', 
            # remove the 'train' prefix.
            # (e.g., 'train/actor_loss' -> 'actor_loss')
            if key.startswith('train'):
                key = key[len('train') + 1:]
                
            # If the metric name starts with 'eval' 
            # remove the 'eval' prefix                
            elif key.startswith('eval'):
                key = key[len('eval') + 1:]
                
            # Replace any remaining '/' values with underscores.
            key = key.replace('/', '_')
            
            # Retrieve the current average value from the meter
            # using value() function defined in AverageMeter() class,
            # and add it to the data dictionary with the processed key.
            data[key] = meter.value()
        
        # Return the dictionary that now contains all metric names 
        # and their current average values.
        return data

    def _remove_old_entries(self, data):
        # Create an empty list to store:
        # - episode number (which training/evaluation episode this 'data' belongs to), 
        # - step number (global step counter in the training process), 
        # - various metrics (e.g., actor loss, episode reward, episode length, etc.), 
        # - time information (e.g., total time of how long training has been running)
        # as a row before saving these information to a .csv file. 
        rows = []
        with self._csv_file_name.open('r') as f:
            # Create a CSV reader to parse each row of the .csv file. 
            reader = csv.DictReader(f)
            
            # Parse each row and if the episode number 
            # in the row is grater than or equal to the 
            # episode number in the new data that is being logged 
            # stop reading and discard that row. 
            # This ensures that if we restart training from an earlier 
            # episode, we remove any existing episodes from 
            # that earlier episodes onwards to avoid duplicate/inconsistent data.
            for row in reader:
                if float(row['episode']) >= data['episode']:
                    break
                rows.append(row)
                  
        with self._csv_file_name.open('w') as f:
            # Create a DictWriter object that will
            # sort the column names
            
                # training:   'step', 'actor_loss', 'total_time', etc.
                
                # evaluation: 'frame', 'step', 'episode', 'episode length', 'episode reward'
                # 'limitation reward', 'total time', etc.
                
            # write dictionaries to the .csv file, and 
            # fill the values of these columns 0 if they are missing.
            writer = csv.DictWriter(f, fieldnames = sorted(data.keys()), restval = 0.0)
            
            # Write the header row (the row that contains all the column names 
            # specified in 'fieldnames' to the csv file.
            writer.writeheader()
            
            # Iterate through each row with episode number less than the 
            # episode number of the current episode and write them back to the file
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        # Check if a .csv writer object has already been created.
        if self._csv_writer is None:
            # If the .csv file does not exist, 
            # set a flag to indicate that the .csv header 
            # (row with column names) should be written.
            # If the .csv file already exists, 
            # remove any rows with episode numbers greater than
            # or equal to the current episode and 
            # set a flag to indicate that the .csv header 
            # should not be written
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            # Open the csv file as append mode so that 
            # new data is added to the end of the file
            # without overwriting existing content.
            self._csv_file = self._csv_file_name.open('a')
            
            # Create a DictWriter object that will
            # sort the column names
            
                # training:   'step', 'actor_loss', 'total_time', etc.
                
                # evaluation: 'frame', 'step', 'episode', 'episode length', 'episode reward'
                # 'limitation reward', 'total time', etc.
                
            # write dictionaries to the .csv file, and 
            # fill the values of these columns 0 if they are missing.
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            
            # If the .csv file does not exist, 
            # write the header row to the .csv file.
            if should_write_header:
                self._csv_writer.writeheader()

        # Write the current 'data' (a dictionary of metric names
        # and values) as a new row in the .csv file.
        self._csv_writer.writerow(data)
        
        # Ensure that any data in the buffer is written to disk 
        # immediately so that the data is saved and can be recovered
        # if the program crashes or if it is interrupted.
        self._csv_file.flush()

    def _format(self, key, value, ty):
        # If the type of the 'value' of the metric ('key') is 'int'
        # convert the 'value' of the metric into 'integer' 
        # and return the string that consists the metric name 
        # and metric value. 
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        
        # If the type of the 'value' of the metric ('key') is 'float' 
        # don't convert it into float because the default value of 
        # the 'value' is already float. 
        # Just return the string that consists the metric name and 
        # metric value.
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        
        # If the type of the 'value' of the metric ('key') is 'time'
        # converts it (assumed to be in seconds) to a datetime.timedelta object
        # and then convert that to a string, 
        # which gives a human-readable format like 
        # "1:23:45" (1 hour, 23 minutes, 45 seconds)
        elif ty == 'time':
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{key}: {value}'
        
        # Otherwise, the type of the 'value' is invalid and 
        # raise an exception that indicates this.
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        # Apply color formatting to prefix (the string that indicates
        # whether this is training or evaluation data). 
        # If we are training the model, show the outputs in yellow color
        # and if we are evaluating the model, show the outputs in green color.
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        
        # During training the model, print the current step, 
        # actor loss, and total time passed since training started in a nice format. 
        # During evaluation of the model, print the current frame, 
        # current step, current episode number, the length of an episode, 
        # the epsiode reward gathered during evaluation, imitation reward 
        # gathered during evaluation, and total time passed 
        # since evaluation started in a nice format.
        # Reminderr: self._formatting = 
        #       COMMON_TRAIN_FORMAT = [('step', 'S', 'int'),
        #                              ('actor_loss', 'L', 'float'),
        #                              ('total_time', 'T', 'time')] for training

        #       COMMON_EVAL_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
        #                             ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
        #                             ('episode_reward', 'R', 'float'),
        #                             ('imitation_reward', 'R_i', 'float'),
        #                             ('total_time', 'T', 'time')] for evaluation.
            
        pieces = [f'| {prefix: <14}']
        # pieces = ['| train          ']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            piece = self._format(disp_key, value, ty)
            pieces.append(piece)
            # pieces = ['| train          ', 'S: 5000', 'L: 0.0214', 'T: 0:45:12']
        print(' | '.join(pieces))
        # | train          | S: 5000 | L: 0.0214 | T: 0:45:12 

    def dump(self, step, prefix):
        # If there is no metric that is logged, the meters dictionary is empty
        # and there is nothing to output and we return early.
        if len(self._meters) == 0:
            return

        # Retrieve and process all the current metric values.
        # This converts the current averages from all metrics
        # into a clean dictionary 'data'
        data = self._prime_meters()
        
        # Add the current step/frame count to the data dictionary
        # that consists of the current averages from all metrics.
        data['frame'] = step
        
        # Write the data dictionary as .csv
        self._dump_to_csv(data)
        
        # Print the metrics in data dictionary in a nice and organized way.
        self._dump_to_console(data, prefix)
        
        # Clear all the metrics, resets their sums and counts to 0.
        self._meters.clear()

class Logger(object):
    def __init__(self, log_dir, use_tb):
        self._log_dir = log_dir
        self._train_mg = MetersGroup(log_dir / 'train.csv', formating = COMMON_TRAIN_FORMAT)
        self._eval_mg = MetersGroup(log_dir / 'eval.csv', formating = COMMON_EVAL_FORMAT)
        
        if use_tb:
            self._sw = SummaryWriter(str(log_dir / 'tb'))
        else:
            self._sw = None

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def log(self, key, value, step):
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value)

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            self.log(f'{ty}/{key}', value, step)

    def dump(self, step, ty=None):
        if ty is None or ty == 'eval':
            self._eval_mg.dump(step, 'eval')
        if ty is None or ty == 'train':
            self._train_mg.dump(step, 'train')

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)

class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        self._logger.log(f'{self._ty}/{key}', value, self._step)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)
