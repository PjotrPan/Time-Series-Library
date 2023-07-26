import os
import xml.etree.ElementTree as ET
import pandas as pd
import yaml
from yaml import CLoader as Loader
from attribute_handler import DatasetCreator

def xml_to_dict(file_name: str) -> dict:
    """Parses a xml file and converts it to a dictionary

    args:
        file_name (string): The full relative path of the xml file

    returns:
        dataset (dict): A python dictionary that holds all the information of the xml files as key, value pairs.
    """
    tree = ET.parse(file_name)
    root = tree.getroot()
    dataset = {attribute.tag: [{key: value for key, value in entry.attrib.items()} for entry in attribute] for attribute in root}
    return dataset

def generate_dataset(xml_file: str = './../dataset/T1DMOhio/raw/540-ws-training.xml', patient_number: int = 540):
    train_test = "test" if "test" in xml_file else "train"
    print("="*60)
    print(f"Generating {train_test} dataset for patient {patient_number}")
    dataset_dict = xml_to_dict(xml_file)
    attributes = ["glucose_level", "finger_stick", "basal", "temp_basal", "bolus",
                  "basis_gsr", "basis_air_temperature", "basis_heart_rate",
                  "basis_skin_temperature", "sleep", "work", "exercise", "meal"]
    dataset = DatasetCreator()
    for attribute in attributes:
        entries = dataset_dict[attribute]
        dataset.add_attribute(attribute, entries)
    return dataset.generate_full_dataset()

def gen_all_datasets():
    for patient_number in [540,544,552,567,584,596,559,563,570,575,588,591]:
        training = generate_dataset(f'./../dataset/T1DMOhio/raw/{patient_number}-ws-training.xml', patient_number)

        train_val_split = int(len(training)*0.75)
        train_ds = training.iloc[:train_val_split, :]
        valid_ds = training.iloc[train_val_split:, :]
        test_ds = generate_dataset(f'./../dataset/T1DMOhio/raw/{patient_number}-ws-testing.xml', patient_number)

        train_ds.to_csv(f"./../dataset/T1DMOhio/single/{patient_number}_train_dataset.csv")
        valid_ds.to_csv(f"./../dataset/T1DMOhio/single/{patient_number}_val_dataset.csv")
        test_ds.to_csv(f"./../dataset/T1DMOhio/single/{patient_number}_test_dataset.csv")

def use_2018_test_data_for_training():
    data_2020 = [540,544,552,567,584,596]
    data_2018 = [559,563,570,575,588,591]

    for patient_number in data_2018:
        training = generate_dataset(f'./../dataset/T1DMOhio/raw/{patient_number}-ws-training.xml', patient_number)
        test_ds = generate_dataset(f'./../dataset/T1DMOhio/raw/{patient_number}-ws-testing.xml', patient_number)

        training = pd.concat((training, test_ds), ignore_index=True, sort=False)
        train_val_split = int(len(training)*0.75)
        train_ds = training.iloc[:train_val_split, :]
        valid_ds = training.iloc[train_val_split:, :]

        train_ds.to_csv(f"./../dataset/T1DMOhio/2020/{patient_number}_train_dataset.csv")
        valid_ds.to_csv(f"./../dataset/T1DMOhio/2020/{patient_number}_val_dataset.csv")


    for patient_number in data_2020:
        training = generate_dataset(f'./../dataset/T1DMOhio/raw/{patient_number}-ws-training.xml', patient_number)

        train_val_split = int(len(training)*0.75)
        train_ds = training.iloc[:train_val_split, :]
        valid_ds = training.iloc[train_val_split:, :]
        test_ds = generate_dataset(f'./../dataset/T1DMOhio/raw/{patient_number}-ws-testing.xml', patient_number)

        train_ds.to_csv(f"./../dataset/T1DMOhio/2020/{patient_number}_train_dataset.csv")
        valid_ds.to_csv(f"./../dataset/T1DMOhio/2020/{patient_number}_val_dataset.csv")
        test_ds.to_csv(f"./../dataset/T1DMOhio/2020/{patient_number}_test_dataset.csv")

def focus_on_one_patient():
    data = [540,544,552,567,584,596,559,563,570,575,588,591]
    for patient_number in data:
        training = generate_dataset(f'./../dataset/T1DMOhio/raw/{patient_number}-ws-training.xml', patient_number)
        test_ds = generate_dataset(f'./../dataset/T1DMOhio/raw/{patient_number}-ws-testing.xml', patient_number)
        training = pd.concat((training, test_ds), ignore_index=True, sort=False)
        training.to_csv(f"./../dataset/T1DMOhio/target/{patient_number}_train_dataset.csv")
        
        vali_ds = generate_dataset(f'./../dataset/T1DMOhio/raw/{patient_number}-ws-training.xml', patient_number)
        vali_ds.to_csv(f"./../dataset/T1DMOhio/target/{patient_number}_val_dataset.csv")

        test_ds.to_csv(f"./../dataset/T1DMOhio/target/{patient_number}_test_dataset.csv")


if __name__ == "__main__":
    use_2018_test_data_for_training()
    focus_on_one_patient()
    gen_all_datasets()