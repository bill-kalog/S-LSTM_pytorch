from torchtext import data


class SSTExample(data.Example):

    @classmethod
    def fromSplitOnFirst(cls, data, fields, field_to_index=None):
        '''
        function specific for loading SST from Harvard
        takes an istance and puts the first character as label
        and the rest as the training point's text
        '''
        ex = cls()
        # field[0] is text and field[1] is label
        a_label, a_text = [data[0], data[2:]]
        a_text = a_text.rstrip('\n')
        # print ('{}-------{}===='.format(a_label, a_text))
        # set text
        setattr(ex, fields[0][0], fields[0][1].preprocess(a_text))
        # set label
        setattr(ex, fields[1][0], fields[1][1].preprocess(a_label))
        return ex
