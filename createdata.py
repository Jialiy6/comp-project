from utils import createdata

if __name__ == '__main__':
    createdata(train_folders=['./data/COCO2014/train2014',
                              './data/COCO2014/val2014'],
               test_folders=['./data/BSD100',
                             './data/Set5',
                             './data/Set14'],
               min_size=100,
               output_folder='./data/')

