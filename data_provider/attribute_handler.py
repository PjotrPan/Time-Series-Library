import pandas as pd
import numpy as np

class DatasetCreator():
    """This class is used to create a pandas Dataframe from a custom selection of attributes

    Attributes:
        attributes: Dictionary which contains the name of an attribute as key
            and the actual attribute object as value.
    """

    def __init__(self, time_interval: int = 300) -> None:
        self.attributes = {}
        self.time_interval = time_interval

    def add_attribute(self, attribute: str, data: dict) -> None:
        """Adds an attribute object to the attributes dictionary based on the attribute name.

        Args:
            attribute (string): Name of the attribute that shall be created.
            data (dict): A dictionary containing all data of one attribute.
        """
        if attribute == "glucose_level": self.attributes.update({attribute: GlucoseLevel(data)})
        elif attribute == "finger_stick": self.attributes.update({attribute: FingerStick(data)})
        elif attribute == "basis_heart_rate": self.attributes.update({attribute: BasisHeartRate(data)})
        elif attribute == "basal": self.attributes.update({attribute: Basal(data)})
        elif attribute == "temp_basal": self.attributes.update({attribute: TempBasal(data, self.time_interval)})
        elif attribute == "bolus": self.attributes.update({attribute: Bolus(data, self.time_interval)})
        elif attribute == "basis_gsr": self.attributes.update({attribute: BasisGSR(data)})
        elif attribute == "basis_air_temperature": self.attributes.update({attribute: BasisAirTemperature(data)})
        elif attribute == "basis_skin_temperature": self.attributes.update({attribute: BasisSkinTemperature(data)})
        elif attribute == "sleep": self.attributes.update({attribute: Sleep(data, self.time_interval)})
        elif attribute == "work": self.attributes.update({attribute: Work(data, self.time_interval)})
        elif attribute == "exercise": self.attributes.update({attribute: Exercise(data, self.time_interval)})
        elif attribute == "meal": self.attributes.update({attribute: Meal(data, self.time_interval)})

    def get_attribute(self, attribute: str) -> "Attribute":
        """Retrives attribute object from dictionary.

        Args:
            attribute (string): Name of the attribute that shall be created.
        
        Returns:
            (Attribute): The attributes object reference.
        """
        return self.attributes[attribute]

    def declare_basetime(self, base_attribute: str) -> np.ndarray:
        """Declares a basetime which is used to merge multiple attribute data batches into one dataset.
        The basetime is a numpy linspace containing time steps in the range of the base attributes
        timestamps array. The time interval sets the step size within the basetime array.

        Args:
            base_attribute (string): Name of the attribute whose timestamps give the start and end time.
            time_interval (int): Time in seconds which describes the difference between two time steps.
        
        Returns:
            basetime (np.ndarray): A numpy linspace representing the basetime
        """
        basetime = self.attributes[base_attribute].timestamps
        basetime = np.linspace(min(basetime), max(basetime), int((max(basetime) - min(basetime))/self.time_interval))
        return basetime

    def transform_to_basetime(self, basetime: np.ndarray, data_batch: np.ndarray, nan_value) -> np.ndarray:
        """Maps the entries of an attributes data batch on the basetime.

        Args:
            basetime (np.ndarray): A numpy linspace representing the basetime
            data_batch (np.ndarray): An array which contains all the data of one attribute.

        Returns:
            data_batch_bt (np.ndarray): Data batch mapped on basetime
        """
        no_of_cols = data_batch.shape[-1]
        indices = [(np.abs(basetime - time)).argmin() for time in data_batch[:,0]]
        data_batch_bt = np.full((len(basetime), max(1,no_of_cols-1)), float(nan_value))
        data_batch_bt[indices] = data_batch[:,1:]

        return data_batch_bt

    def generate_full_dataset(self) -> pd.DataFrame:
        """Generates a dataset by joining all data batches.

        Args:
            configs (dict): Dictionary representing the configs.yaml

        Returns:
            (pd.DataFrame): Pandas Dataframe representing the joined dataset.
        """
        basetime = self.declare_basetime(base_attribute="glucose_level")

        dataset = None
        col_names = []

        for attr in self.attributes.values():
            data_batch = self.transform_to_basetime(basetime, attr.data_batch, attr.nan_value)
            data_batch = attr.interpolate(data_batch)
            if dataset is None:
                dataset = data_batch.astype(np.float)
                col_names.extend(attr.names)
                continue
            dataset = np.concatenate((dataset, data_batch), axis=1)
            col_names.extend(attr.names)

        dataset = pd.DataFrame(dataset, columns=col_names)

        basetime = np.array([pd.to_datetime(t * 1000000000) for t in basetime])
        dataset.insert(0, "datetime", basetime)

        return dataset

class Attribute():
    """Base class for Attributes

    All attributes of the dataset may be classes that inherit from this class. 
    The key idea is to implement the actual data parsing individually for each
    attribute in its own class. Also the dataset can be composed by a custom
    selection of attributes. 

    Attributes:
        timestamp: A list of timestamps as int.
        data_batch: Composite array of all the data for a single attribute.
    """
    def __init__(self) -> None:
        self.timestamps = []
        self.data_batch = None
        self.nan_value = -1

    def process_data(self, data: list) -> None:
        """Parsing the data

        Args:
            data (list): Each element in this list is a dictionary which represents
                    the relevant values for each attribute. Since these dictionaries
                    vary for different attributes the method is implemented in the
                    particular attribute class.
        """
        pass

    def convert_time_to_cos_sin(self):
        """Converts all timestamps of a particular attribute into sine and cosine values
        for minutes and hours. These new arrays are stored in the class variables self.hours
        and self.minutes.

        Currently this function is not used by default. 
        """
        minute_list = [(np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, 60)]
        hour_list = [(np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, 24)]
        pd_datetime = [pd.to_datetime(datetime) for datetime in self.timestamps]
        self.hours = np.array([hour_list[dt.hour] for dt in pd_datetime])
        self.minutes = np.array([minute_list[dt.minute] for dt in pd_datetime])

    def time_string_to_timestamp(self):
        """Converts a list of datetime strings into integer timestamps

        Returns:
            (list): List of Linux timestamps
        """
        return [pd.to_datetime(time_string, dayfirst=True).timestamp() for time_string in self.timestamps]

    def normalize(self, values):
        """Maps any list of values to a range from 0 to 1.
        Type of list entries depends on specific attribute

        Args:
            values (list): List of attribute values 

        Returns:
            (list): List of normalized values
        """
        if len(values) == 0:
            return values
        min_v = min(values)
        max_v = max(values)
        return [(v-min_v)/(max_v-min_v) for v in values]
    
    def interpolate(self, data_batch: np.ndarray) -> np.ndarray:
        return data_batch

class GlucoseLevel(Attribute):
    """Continuous glucose monitoring data, recorded every 5 minutes.

    Attributes:
        timestamps (list(datetime)): Exact timestamp each CGM value was recorded
        glucose_level (list(int)): Glucose level value for each timestamp
    """
    def __init__(self, data) -> None:
        super().__init__()
        self.glucose_level = []
        self.names = ["glucose_level"]
        self.nan_value = np.nan
        self.process_data(data)

    def process_data(self, data):
        for instance in data:
            self.timestamps.append(instance["ts"])
            self.glucose_level.append(int(instance["value"]))
        self.timestamps = self.time_string_to_timestamp()
        self.timestamps = np.expand_dims(self.timestamps, axis=1)

        self.glucose_level = np.array(self.glucose_level)
        self.glucose_level = np.expand_dims(self.glucose_level, axis=1)
        self.data_batch = np.concatenate((self.timestamps, self.glucose_level), axis=1)

class FingerStick(Attribute):
    """Blood glucose values obtained through selfmonitoring by the patient

    Attributes:
        timestamps (list(datetime)): Exact timestamp each time the CGM was measured manually
        finger_stick (list(int)): Glucose level measured manually
    """
    def __init__(self, data) -> None:
        super().__init__()
        self.finger_stick = []
        self.names = ["finger_stick"]
        self.process_data(data)

    def process_data(self, data):
        for instance in data:
            self.timestamps.append(instance["ts"])
            self.finger_stick.append(int(instance["value"]))
        self.timestamps = self.time_string_to_timestamp()
        self.timestamps = np.expand_dims(self.timestamps, axis=1)

        self.finger_stick = np.array(self.finger_stick)
        self.finger_stick = np.expand_dims(self.finger_stick, axis=1)
        self.data_batch = np.concatenate((self.timestamps, self.finger_stick), axis=1)
        
class BasisHeartRate(Attribute):
    """Heart rate, aggregated every 5 minutes. 
    This data is only available for people who wore the Basis Peak sensor band.
    
    Attributes:
        timestamps (list(datetime)): Exact timestamp the heart rate was measured
        heart_rate (list(int)): Heart rate values
    """
    def __init__(self, data) -> None:
        super().__init__()
        self.heart_rate = []
        self.names = ["heart_rate"]
        self.process_data(data)

    def process_data(self, data):
        for instance in data:
            self.timestamps.append(instance["ts"])
            self.heart_rate.append(int(instance["value"]))
        self.timestamps = self.time_string_to_timestamp()
        self.timestamps = np.expand_dims(self.timestamps, axis=1)

        self.heart_rate = np.array(self.heart_rate)
        #self.heart_rate = self.normalize(self.heart_rate)
        self.heart_rate = np.expand_dims(self.heart_rate, axis=1)
        self.data_batch = np.concatenate((self.timestamps, self.heart_rate), axis=1)

class Basal(Attribute):
    def __init__(self, data) -> None:
        super().__init__()
        self.basal_rate = []
        self.names = ["basal_rate"]
        self.process_data(data)

    def process_data(self, data: list) -> None:
        for instance in data:
            self.timestamps.append(instance["ts"])
            self.basal_rate.append(float(instance["value"]))

        self.timestamps = self.time_string_to_timestamp()
        self.timestamps = np.expand_dims(self.timestamps, axis=1)

        self.basal_rate = np.array(self.basal_rate)
        self.basal_rate = np.expand_dims(self.basal_rate, axis=1)
        self.data_batch = np.concatenate((self.timestamps, self.basal_rate), axis=1)
   
    def interpolate(self, data_batch):
        rate = data_batch[0]
        for idx, data in enumerate(data_batch):
            if data <= 0:
                data_batch[idx] = rate
            else:
                rate = data

        return data_batch
    
class TempBasal(Attribute):
    def __init__(self, data, time_interval) -> None:
        super().__init__()
        self.temp_basal_rate = []
        self.names = ["temp_basal_rate"]
        self.time_interval = time_interval
        self.process_data(data)

    def process_data(self, data: list) -> None:
        for instance in data:
            start = pd.to_datetime(instance["ts_begin"], dayfirst=True).timestamp() 
            end = pd.to_datetime(instance["ts_end"], dayfirst=True).timestamp() 
            for timestamp in range(int(start), int(end), self.time_interval):
                self.timestamps.append(timestamp)
                self.temp_basal_rate.append(float(instance["value"]))

        self.timestamps = np.expand_dims(self.timestamps, axis=1)

        self.temp_basal_rate = np.array(self.temp_basal_rate)
        self.temp_basal_rate = np.expand_dims(self.temp_basal_rate, axis=1)
        self.data_batch = np.concatenate((self.timestamps, self.temp_basal_rate), axis=1)

    def interpolate(self, data_batch):
        return data_batch

class Bolus(Attribute):
    def __init__(self, data, time_interval) -> None:
        super().__init__()
        self.bolus = []
        self.names = ["bolus"]
        self.time_interval = time_interval
        self.process_data(data)

    def process_data(self, data: list) -> None:
        for instance in data:
            start = pd.to_datetime(instance["ts_begin"], dayfirst=True).timestamp() 
            end = pd.to_datetime(instance["ts_end"], dayfirst=True).timestamp() 
            for timestamp in range(int(start)-1, int(end), self.time_interval):
                self.timestamps.append(timestamp)
                divisor = 1 if int(start) == int(end) else (int(end) - int(start)) / self.time_interval
                self.bolus.append(float(instance["dose"]) / divisor)

        self.timestamps = np.expand_dims(self.timestamps, axis=1)

        self.bolus = np.array(self.bolus)
        self.bolus = np.expand_dims(self.bolus, axis=1)
        self.data_batch = np.concatenate((self.timestamps, self.bolus), axis=1)

    def interpolate(self, data_batch):
        return super().interpolate(data_batch)
        
class BasisGSR(Attribute):
    """ Galvanic skin response, also known as skin conductance or electrodermal activity. 
    For those who wore the Basis Peak, the data was aggregated every 5 minutes. 
    Despite this attribute’s name, it is also available for those who wore the Empatica Embrace. 
    For these individuals, the data is aggregated every 1 minute
    """
    def __init__(self, data) -> None:
        super().__init__()
        self.gsr = []
        self.names = ["GSR"]
        self.process_data(data)

    def process_data(self, data: list) -> None:
        for instance in data:
            self.timestamps.append(instance["ts"])
            self.gsr.append(float(instance["value"]))

        self.timestamps = self.time_string_to_timestamp()
        self.timestamps = np.expand_dims(self.timestamps, axis=1)

        self.gsr = np.array(self.gsr)
        #self.gsr = self.normalize(self.gsr)
        self.gsr = np.expand_dims(self.gsr, axis=1)
        self.data_batch = np.concatenate((self.timestamps, self.gsr), axis=1)

class BasisAirTemperature(Attribute):
    def __init__(self, data) -> None:
        super().__init__()
        self.air_temparature = []
        self.names = ["air_temp"]
        self.process_data(data)

    def process_data(self, data: list) -> None:
        for instance in data:
            self.timestamps.append(instance["ts"])
            self.air_temparature.append(float(instance["value"]))

        self.timestamps = self.time_string_to_timestamp()
        self.timestamps = np.expand_dims(self.timestamps, axis=1)

        self.air_temparature = np.array(self.air_temparature)
        #self.air_temparature = self.normalize(self.air_temparature)
        self.air_temparature = np.expand_dims(self.air_temparature, axis=1)
        self.data_batch = np.concatenate((self.timestamps, self.air_temparature), axis=1)

class BasisSkinTemperature(Attribute):
    def __init__(self, data) -> None:
        super().__init__()
        self.skin_temparature = []
        self.names = ["skin_temp"]
        self.process_data(data)

    def process_data(self, data: list) -> None:
        for instance in data:
            self.timestamps.append(instance["ts"])
            self.skin_temparature.append(float(instance["value"]))

        self.timestamps = self.time_string_to_timestamp()
        self.timestamps = np.expand_dims(self.timestamps, axis=1)

        self.skin_temparature = np.array(self.skin_temparature)
        #self.skin_temparature = self.normalize(self.skin_temparature)
        self.skin_temparature = np.expand_dims(self.skin_temparature, axis=1)
        self.data_batch = np.concatenate((self.timestamps, self.skin_temparature), axis=1)

class Sleep(Attribute):
    def __init__(self, data, time_interval) -> None:
        super().__init__()
        self.sleep_quality = []
        self.names = ["sleep"]
        self.time_interval = time_interval
        self.process_data(data)

    def process_data(self, data: list) -> None:
        for instance in data:
            start = pd.to_datetime(instance["ts_end"], dayfirst=True).timestamp() 
            end = pd.to_datetime(instance["ts_begin"], dayfirst=True).timestamp() 
            for timestamp in range(int(start), int(end), self.time_interval):
                self.timestamps.append(timestamp)
                self.sleep_quality.append(float(instance["quality"]))

        self.timestamps = np.expand_dims(self.timestamps, axis=1)

        self.sleep_quality = np.array(self.sleep_quality)
        self.sleep_quality = np.expand_dims(self.sleep_quality, axis=1)
        self.data_batch = np.concatenate((self.timestamps, self.sleep_quality), axis=1)

class Work(Attribute):
    def __init__(self, data, time_interval) -> None:
        super().__init__()
        self.work_intensity = []
        self.names = ["work"]
        self.time_interval = time_interval
        self.process_data(data)

    def process_data(self, data: list) -> None:
        for instance in data:
            start = pd.to_datetime(instance["ts_begin"], dayfirst=True).timestamp()
            end = pd.to_datetime(instance["ts_end"], dayfirst=True).timestamp()
            for timestamp in range(int(start), int(end), self.time_interval):
                self.timestamps.append(timestamp)
                self.work_intensity.append(float(instance["intensity"]))

        self.timestamps = np.expand_dims(self.timestamps, axis=1)

        self.work_intensity = np.array(self.work_intensity)
        self.work_intensity = np.expand_dims(self.work_intensity, axis=1)
        self.data_batch = np.concatenate((self.timestamps, self.work_intensity), axis=1)

class Exercise(Attribute):
    def __init__(self, data, time_interval) -> None:
        super().__init__()
        self.exercise_intensity = []
        self.names = ["exercise"]
        self.time_interval = time_interval
        self.process_data(data)

    def process_data(self, data:list) -> None:
        for instance in data:
            start = pd.to_datetime(instance["ts"], dayfirst=True).timestamp()
            end = int(start) + int(instance["duration"])*60
            for timestamp in range(int(start), int(end), self.time_interval):
                self.timestamps.append(timestamp)
                self.exercise_intensity.append(float(instance["intensity"]))
        self.timestamps = np.expand_dims(self.timestamps, axis=1)

        self.exercise_intensity = np.array(self.exercise_intensity)
        self.exercise_intensity = np.expand_dims(self.exercise_intensity, axis=1)
        self.data_batch = np.concatenate((self.timestamps, self.exercise_intensity), axis=1)

class Meal(Attribute):
    def __init__(self, data, time_interval) -> None:
        super().__init__()
        self.carbohydrates = []
        self.names = ["meal"]
        self.time_interval = time_interval
        self.process_data(data)

    def process_data(self, data:list) -> None:
        for instance in data:
            self.timestamps.append(instance["ts"])
            self.carbohydrates.append(float(instance["carbs"]))

        self.timestamps = self.time_string_to_timestamp()
        self.timestamps = np.expand_dims(self.timestamps, axis=1)

        self.carbohydrates = np.array(self.carbohydrates)
        self.carbohydrates = np.expand_dims(self.carbohydrates, axis=1)
        self.data_batch = np.concatenate((self.timestamps, self.carbohydrates), axis=1)