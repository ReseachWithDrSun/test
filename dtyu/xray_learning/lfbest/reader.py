from lfbest import nn_input


file_list_train = [
    '../xray_data/fbb_output/batch-0.bin',
    '../xray_data/fbb_output/batch-1.bin',
    '../xray_data/fbb_output/batch-2.bin',
    '../xray_data/fbb_output/batch-3.bin',
    '../xray_data/fbb_output/batch-4.bin',
    '../xray_data/fbb_output/batch-5.bin',
    '../xray_data/fbb_output/batch-6.bin',
    '../xray_data/fbb_output/batch-7.bin',
    '../xray_data/fbb_output/batch-8.bin'
]
file_list_valid = ['../xray_data/fbb_output/batch-9.bin']


class Reader:
    def __init__(self, batch_size, file_list, num_examples, distort=False):
        self.batch_size = batch_size
        self.file_list = file_list
        self.num_examples = num_examples
        self.distort = distort

    def get_batch(self):
        if self.distort:
            return nn_input.distorted_inputs(self.file_list, self.num_examples,
                                             self.batch_size, shuffle=True)
        else:
            return nn_input.inputs(self.file_list, self.num_examples,
                                   self.batch_size)
