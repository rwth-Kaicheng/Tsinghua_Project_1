import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()
# pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import sounddevice as sd
import wave
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import pyrealsense2 as rs
import os
import time
import whisper
from scipy.io.wavfile import write
import spacy
import noisereduce as nr
import soundfile as sf
from transformers import MarianMTModel, MarianTokenizer

# 预载中译英的模型
# 预载spacy中的单词切割模型
zh2en_model = 'Helsinki-NLP/opus-mt-zh-en'
tokenizer = MarianTokenizer.from_pretrained(zh2en_model)
zh2enmodel = MarianMTModel.from_pretrained(zh2en_model)
nlp = spacy.load("en_core_web_md")

def translate_zh_to_en(text):
    # 编码文本
    encoded_text = tokenizer(text, return_tensors="pt", padding=True)
    # 进行翻译
    translated = zh2enmodel.generate(**encoded_text)
    # 解码翻译结果
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """
        Initializes the Colors class with a palette derived from Ultralytics color scheme, converting hex codes to RGB.

        Colors derived from `hex = matplotlib.colors.TABLEAU_COLORS.values()`.
        """
        hexs = (
            "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
            "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7",
        )#各种预选好的颜色
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Returns color from palette by index `i`, in BGR format if `bgr=True`, else RGB; `i` is an integer index."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hexadecimal color `h` to an RGB tuple (PIL-compatible) with order (R, G, B)."""
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))

#画框的函数
def draw_images(image, boxes, classes, scores, colors, xyxy=True):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[:, :, ::-1])
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    # 初始化字体
    font = ImageFont.truetype(font='configs/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = max((image.size[0] + image.size[1]) // 300, 1)
    draw = ImageDraw.Draw(image)

    # 维护每个类别最高得分的框
    best_boxes = {}
    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        if cls not in best_boxes or best_boxes[cls][1] < score:
            best_boxes[cls] = (box, score, colors[i])

    # 绘制每个类别最高得分的框
    for cls, (box, score, color) in best_boxes.items():
        x1, y1, x2, y2 = box
        label = f'{cls}:{score:.2f}'
        text_origin = (x1, y1 - 10)

        # 绘制边框和标签
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        draw.rectangle([text_origin[0], text_origin[1], text_origin[0] + 100, text_origin[1] + 20], fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)

    return image

#预载glip模型，此文件夹中需要把路径从E盘改为glip_tinyweights
config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = r'E:/premodel/glip_tiny_model_o365_goldg_cc_sbu.pth'

# update the config options with the config file
# manual override some options
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
    show_mask_heatmaps=False
)


def glip_inference(image_, caption_):
    # 为不同类别设置颜色, 从caption提取的类别不同
    colors_ = Colors()

    preds = glip_demo.compute_prediction(image_, caption_)
    top_preds = glip_demo._post_process(preds, threshold=0.3)

    # 从预测结果中提取预测类别,得分和检测框
    labels = top_preds.get_field("labels").tolist()
    scores = top_preds.get_field("scores").tolist()
    boxes = top_preds.bbox.detach().cpu().numpy()

    # 为每个预测类别设置框颜色
    colors = [colors_(idx) for idx in labels]
    # 获得标签数字对应的类别名
    labels_names = glip_demo.get_label_names(labels)

    return boxes, scores, labels_names, colors
    
# 创建 'imgg' 和 'depth' 等文件夹（如果尚不存在）
imgg_path = 'E:/results/img'
depth_path = 'E:/results/depth'
output_path = 'E:/results/pixel'
threed_position_path = 'E:/results/3Dposition'
final_threed_position_path = 'E:/results/final3Dposition'

for path in [imgg_path, depth_path, output_path, threed_position_path, final_threed_position_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# 录音
def record_audio(duration=10, sample_rate=44100, channels=1):
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait()  # Wait until recording is finished
    return recording

def save_audio(file_path, data, sample_rate=44100):
    write(file_path, sample_rate, data)
    print(f"Audio saved to {file_path}")

#利用spacy内置模型执行名词分割
#由于glip生成的名词顺序并不是按照whisper的语序来的，如果要用固定语序那就从录音里把名词抠出，然后重新按语序排列
def extract_nouns(text):
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]  # 提取名词
    return nouns

def process_data_and_filter(final_path, data):
    # 解析数据并按标签和置信度筛选最佳记录
    best_records = {}
    for line in data:
        parts = line.strip().split(',')
        X, Y, Z, width, label, score = parts
        score = float(score)

        # 检查是否为该标签找到更高的置信度
        if label not in best_records or best_records[label]['score'] < score:
            best_records[label] = {'score': score, 'entry': line.strip()}

    # 将最佳记录写入最终文件
    with open(final_path, 'w') as f:
        for record in best_records.values():
            f.write(record['entry'] + '\n')

def main():
    # 录音参数
    duration = 5  # seconds
    sample_rate = 44100  # Sample rate in Hz
    audio_directory = 'E:\\audio'
    if not os.path.exists(audio_directory):
        os.makedirs(audio_directory)

    # 录制环境噪声
    noise_data = record_audio(duration=8)  # 录制5秒的噪声
    noise_file_path = os.path.join(audio_directory, "noise.wav")
    save_audio(noise_file_path, noise_data)

    # 录制主音频
    main_audio_data = record_audio(duration=8)  # 主录音时间可以根据需要设定
    main_audio_file_path = os.path.join(audio_directory, "main_audio.wav")
    save_audio(main_audio_file_path, main_audio_data)
    
    # 读取噪声和主音频
    noise_audio, _ = sf.read(noise_file_path)
    main_audio, sample_rate = sf.read(main_audio_file_path)


    # 应用噪声降噪
    audio_clean = nr.reduce_noise(y=main_audio, sr=sample_rate, y_noise=noise_audio)
    clean_audio_file_path = os.path.join(audio_directory, "clean_audio.wav")
    sf.write(clean_audio_file_path, audio_clean, sample_rate)
    print("Noise reduced audio saved.")


    # 读取whisper模型
    model = whisper.load_model("base")

    # 录音转译
    result = model.transcribe(clean_audio_file_path, language="zh")  # 固定转换中文
    #result = model.transcribe(clean_audio_file_path, language="en") # 固定转换为英文
    
    # 内容中译英
    caption = result["text"]
    print(f"Transcription: {caption}")
    english_translation = translate_zh_to_en(caption)
    print(english_translation)
   

    # 从英文句中提取关键词
    keywords = extract_nouns(english_translation)
    print("Extracted Keywords:", keywords)

    # 构建相机参数
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # RGB相机与深度相机校准
    align_to = rs.stream.color
    alignedFs = rs.align(align_to)

    # 相机内参调用
    profile = pipeline.start(config)
    video_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = video_profile.get_intrinsics()
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy

    try:
        img_count = 0
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = alignedFs.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            boxes, scores, labels_names, colors = glip_inference(color_image, english_translation)

            pil_image = draw_images(color_image, boxes, labels_names, scores, colors)
            image_np = np.array(pil_image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            img_count += 1
            color_img_path = os.path.join(imgg_path, f'{img_count}.jpg')
            depth_img_path = os.path.join(depth_path, f'{img_count}.npy')
            pixel_info_path = os.path.join(output_path, f'{img_count}.txt')
            position_3d_path = os.path.join(threed_position_path, f'{img_count}.txt')

            cv2.imwrite(color_img_path, image_np)
            np.save(depth_img_path, depth_image)

            # 存取像素中心和路径
            with open(pixel_info_path, 'w') as f:
                for box, label_name in zip(boxes, labels_names):
                    center_x = int((box[0] + box[2]) / 2)
                    center_y = int((box[1] + box[3]) / 2)
                    depth_value = depth_image[center_y, center_x]
                    f.write(f'Center Pixel: ({center_x}, {center_y}), Depth: {depth_value}, Label: {label_name}\n')

            best_scores = {}
            for box, label_name, score in zip(boxes, labels_names, scores):
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                depth_value = depth_image[center_y, center_x]
                X = (center_x - cx) * depth_value / fx
                Y = (center_y - cy) * depth_value / fy
                Z = depth_value
                width = np.abs((box[2] - box[0]) / 2)
                if label_name not in best_scores or best_scores[label_name][0] < score:
                    best_scores[label_name] = (score, f'{X}, {Y}, {Z}, {width}, {label_name}, {score}\n')
            with open(position_3d_path, 'w') as f:
                written = set()
                for score_data in best_scores.values():
                    data_string = score_data[1]
               # 检查是否已经写入了相同的分数数据
                    if data_string not in written:
                        f.write(data_string)
                        written.add(data_string)
            
            cv2.imshow("Result", image_np)
            if cv2.waitKey(1) == 27:  # ESC key
                break

            #根据语音语序保存坐标点
            original_text_path = os.path.join(threed_position_path, f'{img_count}.txt')
            final_pos_path = os.path.join('E:\\results\\finalpos', f'{img_count}_reordered.txt')
            with open(original_text_path, 'r') as original_file:
                lines = original_file.readlines()
            labels = [line.strip().split(', ')[-2] for line in lines]
            keyword_indices = {keyword: [] for keyword in keywords}
            for i, label in enumerate(labels):
                for j, keyword in enumerate(keywords):
                    if keyword in label:
                        keyword_indices[keyword].append(i)
            reordered_lines = []
            for keyword in keywords:
                indices = keyword_indices[keyword]
                reordered_lines.extend([lines[idx] for idx in indices])
            with open(final_pos_path, 'w') as final_file:
                final_file.writelines(reordered_lines)

            """ debugging #用于测试句型中单词的词性

            print("Labels:", labels)
            print("Keywords:", keywords)
            print("Keyword Indices:", keyword_indices)

            reordered_lines = []
            for keyword in keywords:
                indices = keyword_indices[keyword]
                reordered_lines.extend([lines[idx] for idx in indices])
                print(f"Reordering {keyword}: Indices {indices}, Lines: {[lines[idx] for idx in indices]}")  # 这行会显示每个关键词和对应的行

            with open(final_pos_path, 'w') as final_file:
                final_file.writelines(reordered_lines)
            """


    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    main()