from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/media/fadhla/Ubunt_2/Image_Captioning/caption_datasets/dataset_coco.json',
                       image_folder='/media/fadhla/Ubunt_2/Image_Captioning/images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/media/fadhla/Ubunt_2/caption_results',
                       max_len=50)