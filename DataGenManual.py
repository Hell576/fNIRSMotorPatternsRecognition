
import os


import scipy.ndimage

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#neDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.
# To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import pandas as pd
import statistics
from scipy.ndimage import convolve



from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display



########################################

def file_toNumPy_array(name: str):
    #numpy.loadtxt or numpy.fromfile

    '''
    Notes from NumPY documentation

Do not rely on the combination of tofile and fromfile for data storage,
 as the binary files generated are not platform independent.
 In particular, no byte-order or data-type information is saved.
 Data can be stored in the platform independent .npy format
 using save and load instead.
 
 The recommended way to store and load data:
     np.save(fname, x)
     np.load(fname + '.npy')
    '''
    dataset = np.fromfile(name, sep='\n')

    return dataset


def file_toPandas_table(name: str):
 #read csv
    table = pd.read_csv(name, delimiter='\t')
    first_entry = table.columns[0]
    table = table.rename({first_entry: 'col1'}, axis=1)

    table.loc[-1] = first_entry
    table.index = table.index + 1
    table.sort_index(inplace=True)

    #print(table)    
    table.describe()
    #type(table['col1'][0])
    table['col1'][0].split(' ')
    for i in range(len(table)):
        table.loc[i, 'col1'] = list(map(float, table.loc[i, 'col1'].split(' ')))
      #table['col1'][i] = list(map(float, table['col1'][i].split(' ')))

    #print(table)
    table.describe()

    '''
    #[DataStructure]: columns = S-D-Key
    #S-D-Key="1-1:1,1-2:2,1-3:3,1-4:4, ... ,1-8:8,
              2-1:9,2-2:10,2-3:11,2-4:12, ... ,2-8:16,
              3-1:17,3-2:18, ... 
         ... ,8-1:57,8-2:58,8-3:59,8-4:60,8-5:61,8-6:62,8-7:63,8-8:64,"
    '''
    k = 1
    columns = []
    for i in range (1, 8+1):
        for j in range (1, 8+1):
            columns.append('{}-{}:{}'.format(i,j,k))
            k = k + 1

    res_table = pd.DataFrame(table['col1'].tolist(), columns=columns, index=table.index)
    return(res_table)

#######################
    #for item in flowers_root.glob("*"):
    #    print(item.name)



def find_EvtMarkers(evt_filename: str) -> np.ndarray:
    with open(evt_filename, 'r') as f:

        #Find '[Markers]' line
        line = f.readline()
        num_line = 1
        while line != '' and line.find('[Markers]') == -1:
            line = f.readline()
            num_line = num_line + 1

        num_line = num_line + 1

        tbl_label = f.readline()
        line = f.readline()
        num_line = num_line + 2

        evtLine_tbl = np.fromstring(line, sep=' ')


        while True:
            line = f.readline()
            if line != '\n' and line.find('#"') == -1:
                evtLine_tbl = np.vstack( (evtLine_tbl, \
                                          np.fromstring(line, sep=' ')) )

            else: break

            num_line = num_line + 1


    return evtLine_tbl


def cutData_by_evtLine(Hb_matrix:np.ndarray, evt_filename: str, classCount=5) -> list: #of ReggedTensors

    evtLine_tbl = find_EvtMarkers(evt_filename)
    len_evt = evtLine_tbl.shape[0]#len(evtLine_tbl)

    row_splits = [int(evtLine_tbl[i][2])-1 for i in range(len_evt)]  #WE NEED to set natural indexline that will be equal to length of evtLine_tbl (that means count from 1, but not from zero as in arrays),
    RT_row_splits = [0] + row_splits + [len(Hb_matrix)]                    #cause this requires RT_row_splits <==> row_splits for tf.RaggedTensor.from_row_splits() func
    splitted_arrs = tf.RaggedTensor.from_row_splits(values=Hb_matrix, row_splits=RT_row_splits)     ###length of splitted_arrs is 22, len of evtLine_tbl is 21. That's wright, but event in file before 19th line is odd

    #Grouping data according to classes = evtLine_tbl[i][1]
    # [[False] * len(evtLine_tbl)] * 5 - forbidden - the same adress
    bm = [[False for i in range(len_evt)] for j in range(classCount)] #boolean mask


    for i in range(len_evt):
        #print('i=', i)
        event_index = int(evtLine_tbl[i][1])
        # match-case statement available since python 3.10. Code was written in 3.12
        if event_index-1 in range(classCount) and event_index == 1:
            bm[0][i] = True
            continue
        elif event_index-1 in range(classCount) and event_index == 2:
            bm[1][i] = True
            continue
        elif event_index-1 in range(classCount) and event_index == 3:
            bm[2][i] = True
            continue
        elif event_index-1 in range(classCount) and event_index == 4:
            bm[3][i] = True
            continue
        elif event_index-1 in range(classCount) and event_index == 5:
            bm[4][i] = True
            continue

    # DONT FORGET TO DELETE 0st elem of each commDataset:   Relax_c, phGrip_c,

    for i in range(len(bm)):
        bm[i].insert(0, False) #removing first evt from 0 upto 18 include line

    #UNiversum classCount = 5, it could be for instance only phGrip and mentUngrip in file, but it will work, cause all the rest gives [], which will be removed in dataset
    Relax_spec = tf.ragged.boolean_mask(splitted_arrs, mask=bm[0])
    phGrip_spec = tf.ragged.boolean_mask(splitted_arrs, mask=bm[1])
    phUngrip_spec = tf.ragged.boolean_mask(splitted_arrs, mask=bm[2])
    mentGrip_spec = tf.ragged.boolean_mask(splitted_arrs, mask=bm[3])
    mentUngrip_spec = tf.ragged.boolean_mask(splitted_arrs, mask=bm[4])    #tf.RaggedTensors

    #DO NOT CHANGE!!! NO REDUCE!!!! ORDER DEPENDENT LABELS
    return [Relax_spec, phGrip_spec, phUngrip_spec, mentGrip_spec, mentUngrip_spec]#actions_ds

def make1Dim(matrix:np.ndarray) -> np.array:
    kernel = np.ones((matrix.shape[0], 1))
    #print(kernel.shape)
    conv_array = scipy.ndimage.convolve(matrix, kernel)
    array1D = conv_array[:, 1]
    # print(array1D[:, 1].shape)
    return array1D
def splitToTensors(HbO_file:str, HbR_file:str, \
                          hdr_path:str, labels, equal_len=False, \
                          elem_type='listOfRaggedTensorsOfTensors',
                          reduceClass=[-1]) -> list: #of RaqggedTensors or Tensors if equal_len=True

        HbO = file_toPandas_table(HbO_file)#file_toNumPy_array(HbO_file) #
        HbR = file_toPandas_table(HbR_file)
        Hb_array = np.append(HbO, HbR, axis=1) #and split

        #to one_dim
        #Hb_array1D = make1Dim(Hb_array)
        #print(Hb_array1D.shape)

        #Metodika podgotovki dannyx na obuchenie #axis=0 == arrow down | axis=1 == ->
        #Hb = np.append(HbO, HbR, axis=0) #and test and valid ds are separate pair of files
        # both ds by axis=1 and axis=0 launch to relearn
        actionsRTList = cutData_by_evtLine(Hb_array, hdr_path, len(labels)) #as actions list Of RaggedTensors Of Tensors

        '''
        if equal_len == True:
            raggedOfRagged = []
            for i in range(len(actions_ds)):
                actions_ds[i] = actions_ds[i].to_tensor() #from RaggedTensor
        '''

        # here we convert set of RaggedTensors of Tensors into a set of Lists of Tensors. Set is a list()
        # it is need for equalizing, raggedTensors were comfortable to split unequal data
        if elem_type == 'listOfLists': #as actions list Of Lists Of Tensors
            listOfListsOfTensors = []
            for i in range(len(actionsRTList)):
                listOfListsOfTensors.append(list())
                #COUNT DEP
                if (i+1 not in set(reduceClass)):
                    #print('list i:', listOfListsOfTensors[i]) #.to_list()
                    for j in range(actionsRTList[i].shape[0]):
                        listOfListsOfTensors[i].append(actionsRTList[i][j])

            return listOfListsOfTensors

        return actionsRTList

def train_valid_testDivision(numpyArray):
    #we'll divide basic dataset Hb into 3 subdatasets: train_ds, validation_ds, test_ds. Read more about division datasets before ML, if you don't find out why  
    #train will be 75% of basic dataset (train_splitVal) and other valid_ds and test_ds will be (100% - 75%) / 2 -> validntest_splitVal
    train_splitInd = int(len(numpyArray) * 0.8)#int(len(Hb) * .75) # =train_splitVal
    validntest_splitVal = (1 - 0.8) / 2
    #tf.ragged.to_tensor

    train_ds = numpyArray[ : train_splitInd]#tf.convert_to_tensor(Hb[ : train_splitInd]) #preprocess_dataset(train_files)


    #here is the border that highlights the end of valid_dataset
    valid_splitInd = int(train_splitInd + len(numpyArray)*validntest_splitVal)


    val_ds = numpyArray[train_splitInd: valid_splitInd]#tf.convert_to_tensor(Hb[train_splitInd: valid_splitInd])    #preprocess_dataset(val_files)


    #here is the border that highlights the end of test_dataset
    test_splitInd = int(valid_splitInd + len(numpyArray)*validntest_splitVal)


    test_ds = numpyArray[valid_splitInd: test_splitInd]#tf.convert_to_tensor(Hb[valid_splitInd: test_splitInd])    #preprocess_dataset(test_files)

    return train_ds, val_ds, test_ds

def equalizeAllTensorsLength(actionsList: list, force_value=-1) -> list:
    #for fileLists as list of lists of Tensors #if type(actions_s[0]) == list
    if (force_value <= -1):
        #find common len to all tensors
        listMaxShapeLens = [np.max([actionsList[i][j].shape[0] for j in range(len(actionsList[i]))])
                            for i in range(len(actionsList))]
        #print('shapes len: ', listMaxShapeLens)
        listMaxShapeLens = np.unique(listMaxShapeLens).tolist()
        listMaxShapeLens.sort()
        commonShapeLen = statistics.median_low(listMaxShapeLens)

    else: commonShapeLen = force_value
    #print('shapes len: ', listMaxShapeLens)
    #print('com shlan: ', commonShapeLen)
    listOfListsOfTensors = actionsList

    #equalizing length to common len
    for i in range(len(listOfListsOfTensors)):
        listOfTensors = listOfListsOfTensors[i]
        for j in range(len(listOfTensors)):
            tensorShape = listOfTensors[j].shape

            if commonShapeLen > tensorShape[0]:
                zero_padding = tf.zeros([commonShapeLen - tensorShape[0], tensorShape[1]], dtype=tf.float64)
                listOfTensors[j] = tf.concat( [listOfTensors[j], zero_padding], axis=0)
            elif commonShapeLen < tensorShape[0]:
                listOfTensors[j] = tf.slice(listOfTensors[j], [0,0], [commonShapeLen,tensorShape[1]])
            # if commonShapeLen == tensorShape[0] then we cant concat (0,)-shape tens with tensor shape (?, 128)

        # print('new_tens: ', listOfTensors[j])
        listOfListsOfTensors[i] = listOfTensors

    #for i in range(len(listOfListsOfTensors)):
    #    print('list',i, listOfListsOfTensors[i])

    '''
       #for actions_ds as list of RaggedTensors of Tensors #if type(actions_s[0]) == tf.RaggedTensor
       listOfRTensorsOfTensors = actions_ds
       listOfListsOfTensor = []
       for i in range(len(actions_ds)):
           listOfListsOfTensors.append(listOfRTensorsOfTensors[i].to_list())
       tf.ragged.stack()
    '''

    return listOfListsOfTensors



#Add dimension
def addDimension(actionsList:list) -> list: #of lists of Tensors

    for i in range(len(actionsList)):
        for j in range(len(actionsList[i])):
            actionsList[i][j] = tf.expand_dims(actionsList[i][j], axis=-1)

    #for i in range(len(actionsList)):
    #    print('list',i, actionsList[i])
    return actionsList
    '''
    listOfListsOfTensors = []

    for i in range(len(fileLists)):

        listOfListsOfTensors.append(list())
        for j in range(fileLists[i].shape[0]):
            listOfListsOfTensors[i].append(tf.expand_dims(fileLists[i][j], axis=-1))    #analog operation tf.reshape()
            #print('raggedt_el',i,j,':', listOfListsOfTensors[i][j]) #tf.ragged.stack

    print(type(listOfListsOfTensors))

    return listOfListsOfTensors
    '''


def preprocessForNN(actionsList:list) -> list:
    #it equalizes 2nd shape
    actionsList = equalizeAllTensorsLength(actionsList, force_value=117)
    #tf.squeeze(audio, axis=-1)
    '''
    for i in range(len(actionsList)):
        for j in range(len(actionsList[i])):
            # transposing
            #actionsList[i][j] = tf.transpose(actionsList[i][j])
            # Convert the waveform to a spectrogram via a STFT.
            
            #spectrogram = tf.signal.stft(
            #    actionsList[i][j],
            #    frame_length=actionsList[i][j].shape[1],
            #    3)  #255, y = 1-3 or frame_step=actionsList[i][j].shape[0]
            # Obtain the magnitude of the STFT.
            #spectrogram = tf.abs(spectrogram)
            # print('Spectrogram shape before:', spectrogram.shape)
            #actionsList[i][j] = spectrogram
            
     '''
    #actionsList = addDimension(actionsList)
    return actionsList

'''
def encodeLabelForNN(label: tf.Tensor, commands:list):
    #str labels arent available for nn, only digits
    label = label.numpy().decode('utf-8')
    print('lab', label)
    label_id = tf.argmax(label == commands)
    print('lid', label_id)
    label = label_id

    return label
'''
def make_oneCommand_dataset(OneCommandTimeframes,      #list of Tensors or Tensor of Tensors
                            label):

    samplesCount = len(OneCommandTimeframes)    #.shape[0]
    #timeframe_lines = [tf.Tensor for i in range(samplesCount)]
    #label_tensors = ['' for i in range(samplesCount)] #tf.Tensor can be instead of ''
    datasetLines = [tf.data.Dataset for i in range(samplesCount)]

    for i in range(samplesCount):
        timeframe_dsline = tf.data.Dataset.from_tensors(OneCommandTimeframes[i])

        #label_tensor = tf.constant(label, dtype=tf.string)
        #label_tensor.numpy().decode('utf-8')
        #label_tensor = encodeLabelForNN(label_tensor, commands)

        label_tensor = tf.constant(label, dtype=tf.int32)
        label_dsline = tf.data.Dataset.from_tensors(label_tensor)
        datasetLines[i] = tf.data.Dataset.zip(timeframe_dsline, label_dsline)

    oneCommandDs = tf.data.Dataset.choose_from_datasets(datasetLines, tf.data.Dataset.range(samplesCount))

    #for line in oneCommandDs.take(5):
    #    print('take ', line[0].numpy())

    return oneCommandDs

def makeDsesByLabels(allTimeframes, commands, save=False, path='') -> list:
    sub_dsesList = []
    for i in range(len(commands)):
        sub_dsesList.append(make_oneCommand_dataset(allTimeframes[i], (i) )) #commands.get(i+1) #(i)

    if (save == True):
        if path != '':
            for i in range(len(sub_dsesList)):
                sub_dsesList[i].save(path + str(i))
        else:
            print('Save failure: no path for ds files specified. Name at least folder name inside proj code directory')


    return sub_dsesList

def uniteDatasetsInOne(sub_dses: list) -> tf.data.Dataset:

    ds = sub_dses[0]
    for i in range(1, len(sub_dses)):
        ds = ds.concatenate(sub_dses[i])

    #ds = tf.data.Dataset.choose_from_datasets(sub_dses, tf.data.Dataset.range(21))
    #ds = ds.map(lambda x,y: [x,y])

    return ds

def loadDatasetsList(filenameWithoutPtNum: str, parts_num=5) -> list:
    listOfDatasets = []
    for i in range(parts_num):
        listOfDatasets.append(tf.data.Dataset.load(filenameWithoutPtNum + str(i)))

    return listOfDatasets

def make1DsLine(timeframeSample: tf.Tensor, label) -> tf.data.Dataset:

    timeframe_dsline = tf.data.Dataset.from_tensors(timeframeSample)

    label_tensor = tf.constant(label, dtype=tf.int32)
    label_dsline = tf.data.Dataset.from_tensors(label_tensor)

    datasetLine = tf.data.Dataset.zip(timeframe_dsline, label_dsline)
    return datasetLine

def gatherLinesToDs(groups: list, commands:dict) -> tf.data.Dataset:
    sub_dsesList = []
    class_nums = []
#POSITION DEP
    for k in commands.keys():
        class_nums.append(k-1) #k-1 #POS DEP
    #class_nums = tf.one_hot(class_nums, len(class_nums))
    #print(len(class_nums) == len(groups))

    samplesCount = 0
    for i in range(len(groups)):
        group = groups[i]
        group_label = class_nums[i]

        for j in range(len(group)):
            sub_dsesList.append(make1DsLine(group[j], group_label))
            samplesCount = samplesCount + 1

    fullDs = tf.data.Dataset.choose_from_datasets(sub_dsesList, tf.data.Dataset.range(samplesCount))
    return fullDs

#No SUBDS to save
def trioFilesToDataset(HbO_file: str, HbR_file: str, \
            hdr_file: str, labels:dict, equalTenslen = True, reduceClass=[-1]) -> tf.data.Dataset:  #*HbO_files, *HbR_files

    elem_type = 'listOfLists'# or 'listOfRaggedTensorsOfTensors'
    tensNestedList = splitToTensors(HbO_file, HbR_file, hdr_file,
                                    labels, equalTenslen, elem_type,
                                    reduceClass)
    tensNestedList = preprocessForNN(tensNestedList)

    ds = gatherLinesToDs(tensNestedList, labels)
    #Kosttil' izza togo chto posle concatenate oper net dostupa k datasetu
    #ds.save('TempDS' + str(descr_number))
    #ds = tf.data.Dataset.load('TempDS')
    #shutil.rmtree('TempDS')#os.rmdir('TempDS')

    return ds

def findRestWl2HdrFiles(HbO_fname: str):
    dot_index = HbO_fname.rfind(".")  # find the last index
    HbR_fname = tf.io.gfile.glob(HbO_fname[0:dot_index] + ".wl2")[0]
    hdrFname = tf.io.gfile.glob(HbO_fname[0:dot_index] + ".hdr")[0]

    return HbR_fname, hdrFname
def massDataset(commands, wl1_1stFilename:str, wl1Filenames=[], reduceClass=[]) -> tf.data.Dataset:
    #delete sub_dsX folders after program finishes its work
    wl2_1stFilename, hdr_1stFilename = findRestWl2HdrFiles(wl1_1stFilename)
    #make sure that every file is not empty
    ds = trioFilesToDataset(wl1_1stFilename, wl2_1stFilename, hdr_1stFilename, commands, reduceClass=reduceClass)

    for i in range(len(wl1Filenames)):
        HbO_fname = wl1Filenames[i]
        HbR_fname, hdrFname = findRestWl2HdrFiles(HbO_fname)

        if (HbO_fname == [] or HbR_fname == [] or hdrFname == []):
            continue

        ds1 = trioFilesToDataset(HbO_fname, HbR_fname, hdrFname, commands, reduceClass=reduceClass)
        ds = ds.concatenate(ds1)


    return ds

def displayDataset(ds:tf.data.Dataset, dsname=''):
    # Dataset inside
    k = 0
    for elem in ds:  # .as_numpy_iterator():
        print(dsname + ' ds el', k, ': ', elem)
        k = k + 1
########################################
##############################  MAIN CODE BEGIN  ####################################
#tf.compat.v1.enable_eager_execution()
AUTOTUNE = tf.data.AUTOTUNE

#need to 'pip install pypiwin32' first
import tkfilebrowser as tkfb

def gen_ds_script():
    print('Выберите путь к файлам данных')
    pathToCreate = tkfb.askopendirname(title='Выберите путь к файлам данных',
                                       initialdir=experiment_dir,
                                       okbuttontext='Choose path',
                                       foldercreation=True)

    if (pathToCreate == ''):
        print('Нажали "Отмена" или не найдено ни одного wl1 файла (значит нет и wl2, hdr) ')
        return -1
    print(pathToCreate)
    HbO_filenames = tf.io.gfile.glob(pathToCreate + "/*.wl1")
    totalCount = len(HbO_filenames)
    print('В директории посчитано ', totalCount, ' wl1 файлов')
    print('(и мы рассчитываем на столько же wl2 и hdr одонимённых файлов)')

    print(
        'Укажите сколько файлов из посчитанных для wl1 всего будет задействовано в train_ds, valid_ds, test_ds и sample_ds вместе взятых')
    needCount = int(input())
    if needCount < totalCount:
        HbO_filenames = HbO_filenames[:needCount]
    else:
        print('Вы нажали Enter или указали недействительное число, превышающее число всех файлов wl1')
        print('Поэтому выбрано максимальное кол-во')

    sampHbO_filename = HbO_filenames[len(HbO_filenames) - 1]

    HbO_filenames = HbO_filenames[:len(HbO_filenames) - 1]

    # Nomera sobity
    commands = {1: b'Relaxation', \
                2: b'Physical_grip', \
                3: b'Physical_ungrip', \
                4: b'Mental_grip', \
                5: b'Mental_ungrip'}

    print('Укажите через пробел номера событий по порядку, которые нужно исключить из датасета')
    classesToReduce = input().split(sep=' ')
    classesToReduce = sorted(map(int, classesToReduce))

    ds = massDataset(commands, HbO_filenames[0],
                     HbO_filenames[1:], reduceClass=classesToReduce)

    # displayDataset(ds, 'full_ds')

    hbO_sampleFile = sampHbO_filename
    hbR_sampleFile, hdr_sampleFile = findRestWl2HdrFiles(hbO_sampleFile)

    sample_ds = trioFilesToDataset(hbO_sampleFile, hbR_sampleFile, hdr_sampleFile, commands)  # reduceClass=[1,2]

    # ds = ds.map(lambda x, y: (tf.transpose(x), y))

    comkeys = list(commands.keys())
    classLabeling = set(comkeys) - set(classesToReduce)
    joined_str = ''.join([str(el) for el in classLabeling])

    addname = input('Укажите добавочное имя датасету по желанию или нажать enter')

    fulldsnamepath = 'Full' + joined_str + 'DsLSTM' + addname
    ds.save(fulldsnamepath)
    sampdsnamepath = 'sample' + joined_str + 'DsLSTM' #+ addname
    sample_ds.save(sampdsnamepath)
    print('Датасет сохранён в директории, откуда запущена была пррграмма c именем ' + fulldsnamepath)
    print('sample_ds сохранён туда же с именем ' + sampdsnamepath)



if __name__ == "__main__":
    experiment_dir = 'nowrapDataset'

    while (True):
        print('Что вы хотите сделать?')
        print('1 - сгенерировать датасет')
        print('2 - cделать конкатенацию 2х датасетов в 1(без удаления исходных)')
        print('0 - закрыть программу')
        choose = input()
        if choose == '1':
            gen_ds_script()
        elif choose == '2':
            print('зажимайте и удерживайте ctrl потом кликайте на 2 папки')
            pathesToCreate = tkfb.askopendirnames(title='Выберите путь к 2м папкам датасета',
                                                  initialdir='N:\\Users\\MAXUS\\PycharmProjects\\timestamp311',
                                                  okbuttontext='Choose folder(s)',
                                                  foldercreation=True)

            if len(pathesToCreate) < 2:
                print('NIZYA TAK')
                secondPath = tkfb.askopendirname(title='Ещё один путь к 2му датасету нужен',
                                                           initialdir='N:\\Users\\MAXUS\\PycharmProjects\\timestamp311',
                                                           okbuttontext='Choose folder(s)',
                                                           foldercreation=True)
            else: secondPath = pathesToCreate[1]
            ds1 = tf.data.Dataset.load(pathesToCreate[0])
            ds2 = tf.data.Dataset.load(secondPath)
            result_ds = uniteDatasetsInOne([ds1, ds2])
            #concsavepath = ''.join(filter(str.isdigit, ds1_shortname + ds2_shortname))
            addname = input('Укажите добавочное имя датасету по желанию или нажать enter')

            result_ds.save('concatDS' + addname)
            print('Сохранено как concatDS')

        elif choose == '0':
            break
        else:
            print('Не понял')
            continue

        #print('Делать конкатенацию этого датасета со следующим?')



