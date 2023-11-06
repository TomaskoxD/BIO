import pandas as pd
import numpy as np

class Saver:
    """
    Saver

    Class to save the results of the model. It saves the results generated by the model in a csv file.

    Args:
        label_list: List of labels.

    Returns:
        None.
    """
    def __init__(self, label_list):
        self.label_list = label_list
        self.n_class = len(label_list)

    def save(self, label_test_file, dataPRED, save_file):
        datanpPRED = np.squeeze(dataPRED.cpu().numpy())
        df_tmp = pd.read_csv(label_test_file)
        image_names = df_tmp["image"].tolist()

        result = {self.label_list[i]: datanpPRED[:, i] for i in range(self.n_class)}
        result['image_name'] = image_names
        out_df = pd.DataFrame(result)

        name_order = ['image_name'] + self.label_list
        out_df.to_csv(save_file, columns=name_order)