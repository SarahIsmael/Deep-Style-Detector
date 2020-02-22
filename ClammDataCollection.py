
import os, sys
import shutil
import argparse

def parse_commandline_args():
    
    parser = argparse.ArgumentParser(description='Deep Learning Model')
    #image processing parameters
    parser.add_argument('--images_dir_training', default = 'ICDAR2017_CLaMM_Training', help='')
    parser.add_argument('--images_dir_test', default = 'ICDAR2017_CLaMM_task1_task3', help='')
    
    parser.add_argument('--csv_training', default = "ICDAR2017_CLaMM_Training/0_info_ICDAR2017_CLaMM_Training.csv", help='csv training f')
    parser.add_argument('--csv_test', default = "ICDAR2017_CLaMM_task1_task3/0_info_CDAR2017_CLaMM_task1_task3.csv", help='csv test f')
    return parser.parse_args(sys.argv[1:])

args= parse_commandline_args()


def get_labels(csv_file, index_filename, index_script_type):
    input_file = open(csv_file,'r')
    header = input_file.readline()
    input_lines = input_file.readlines()
    dic_label = {}
    for i, line in enumerate(input_lines):
        s = line.strip().split(';')
        if len(s)<2:
            continue
        image_name=s[index_filename]
        image_style=int(s[index_script_type])-1
        dic_label[image_name]=image_style
    return dic_label

if __name__ =='__main__':
    
    values_labels= {0: 'Caroline', 1 :'Cursiva', 2:'Half', 3:'Humanistic', 
                   4:'HumanisticCursive', 5:'Hybrida', 6:'Praegothica', 7:'Semihybrida', 
                   8:'Semitextualis', 9:'SouthernTextualis', 10:'Textualis', 11:'Uncial'
                   }

    if not os.path.isdir(args.images_dir_training+'_labeled'):
        os.mkdir(args.images_dir_training+'_labeled')
    if not os.path.isdir(args.images_dir_test+'_labeled'):
        os.mkdir(args.images_dir_test+'_labeled')
    
    "add labels to files in the training folder"
    image_file_names_train = os.listdir(args.images_dir_training)
    dic_label_train=get_labels(args.csv_training, index_filename=0, index_script_type=1)
    for i, image_name_train in enumerate(image_file_names_train):
        if not image_name_train.endswith('.tif'):
            continue
        style = values_labels[dic_label_train[image_name_train]]
        shutil.copyfile(args.images_dir_training+'/'+image_name_train, 
                    args.images_dir_training+'_labeled'+'/'+style+'-'+image_name_train)
    
    "add labels to files in the test folder"
    image_file_names_test = os.listdir(args.images_dir_test)
    dic_label_test=get_labels(args.csv_test, index_filename=1, index_script_type=2)
    for i, image_name_test in enumerate(image_file_names_test):
        if not image_name_test.endswith('.tif'):
            continue
        style = values_labels[dic_label_test[image_name_test]]
        shutil.copyfile(args.images_dir_test+'/'+image_name_test, 
                    args.images_dir_test+'_labeled'+'/'+style+'-'+image_name_test)
    
    