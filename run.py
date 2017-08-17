import class_model as cm

# # Directories and constants
train_dir = 'data/training/'
test_dir = 'data/test_1/'

# # Sliding Window size
winW = 64
winH = 64

#run
im,rank_templates,rank_labels,suit_templates,suit_labels=cm.open_bounding_boxes(train_dir)
test_im,_ =cm.get_test_imgs(test_dir)
results = cm.classification(test_im,rank_templates,rank_labels,suit_templates,suit_labels)
print(results)
