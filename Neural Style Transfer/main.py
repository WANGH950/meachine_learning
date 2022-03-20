import model.NST as model
from PIL import Image
import matplotlib.pyplot as plt

alpha = 1
beta = 1000
gamma = 1000
epoch = 1500
style_features_name = {
    '0': 'conv1_1',
    '5': 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '28': 'conv5_1'
}
content_feature_name = {
    '30': 'conv5_2'
}
style_weights = {
    'conv1_1':0.2,
    'conv2_1':0.2,
    'conv3_1':0.3,
    'conv4_1':0.4,
    'conv5_1':0.5
}

if __name__ == '__main__':
    print('Loading images...')
    content_path = input('Please input content image file name: ')
    content_image = Image.open('./images/content_image/'+content_path)
    style_path = input('Please input style image file name: ')
    style_image = Image.open('./images/style_image/'+style_path)
    output_path = input('Please input output file name: ')
    print('Creating model...')
    # 实例化网络
    nst = model.NST(
        style_image=style_image,
        content_image=content_image,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        style_features_name=style_features_name,
        content_feature_name=content_feature_name,
        style_weights=style_weights
    ).cuda()
    print('Training...')
    # 调用训练器训练
    model.NST.train(nst,epoch)
    image = nst.get_image()
    image.show()
    image.save('./outputs/'+output_path)
    print("Image saved to './outputs/"+output_path+"'")