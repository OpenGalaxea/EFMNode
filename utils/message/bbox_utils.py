from google import genai
from google.genai import types
import re
import cv2 as cv
import time
import numpy as np

MODEL_ID = "gemini-robotics-er-1.5-preview"
API_KEY = ""
client = genai.Client(api_key=API_KEY)


prompt_template = """
      The robot is asked to {instruction}. Return bounding box of the first required interaction 
      object as a JSON array with labels. Only return bbox with the max likelihood. Never return masks or code fencing. 
      The format should be as follows: [{"box_2d": [ymin, xmin, ymax, xmax],
      "label": <label for the object>}] normalized to 0-1000. The values in
      box_2d must only be integers
      """

def retry(func, max_retries=3):
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(2)
        raise Exception(f"All {max_retries} attempts failed")
    return wrapper


def simple_visual_bbox(image_array, bbox):
    x1, y1, x2, y2 = bbox
    vis_image = image_array.copy()
    cv.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imwrite("gemini_debug_bbox.jpg", vis_image)

def get_simple_vb_imgcv(image_array, bbox):
    x1, y1, x2, y2 = bbox
    vis_image = image_array.copy()
    cv.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return vis_image

@retry
def call_gemini_for_bbox(image_array, instruction):
    image_array = cv.cvtColor(image_array, cv.COLOR_RGB2BGR)
    h, w, _ = image_array.shape
    _, image_bytes = cv.imencode('.jpg', image_array)
    image_bytes = image_bytes.tobytes()
    prompt = prompt_template.replace("{instruction}", instruction)
    start_time = time.time()
    print("start calling gemini, waiting...")
    image_response = client.models.generate_content(
      model=MODEL_ID,
      contents=[
        types.Part.from_bytes(
          data=image_bytes,
          mime_type='image/jpeg',
        ),
        prompt
      ],
      config = types.GenerateContentConfig(
          temperature=0.5,
          thinking_config=types.ThinkingConfig(thinking_budget=5)
      )
    )
    print(f"gemini inference time: {time.time() - start_time} seconds")
    bbox = image_response.text
    print(bbox)
    bbox = re.findall(r'\{"box_2d": \[(\d+), (\d+), (\d+), (\d+)\], "label": "([^"]+)"\}', bbox)[0]
    ymin, xmin, ymax, xmax, label = bbox
    scaled_bboxes = [
        int(int(xmin) / 1000 * w),
        int(int(ymin) / 1000 * h),
        int(int(xmax) / 1000 * w),
        int(int(ymax) / 1000 * h),
    ]
    print(f"xmin: {scaled_bboxes[0]}, y_min: {scaled_bboxes[1]}, \
            x_max: {scaled_bboxes[2]}, y_max: {scaled_bboxes[3]}")
    simple_visual_bbox(image_array, scaled_bboxes)
    return scaled_bboxes

def get_paligemma_box_instruction(image, bbox, target_image_size=224, scale=1024):
    bbox = np.array(bbox)
    h, w  = image.shape[:2]
    h_scale, w_scale = target_image_size / h, target_image_size / w
    bbox = bbox * np.array([w_scale, h_scale, w_scale, h_scale])
    image = cv.resize(image, (target_image_size, target_image_size))
    simple_visual_bbox(cv.cvtColor(image, cv.COLOR_RGB2BGR), bbox) # simple resize for visualization here
    bbox = np.clip(np.round(bbox / target_image_size * scale), 0, scale - 1).astype(np.int32)
    rel_x1, rel_y1, rel_x2, rel_y2 = bbox
    y_min = min(rel_y1, rel_y2)
    x_min = min(rel_x1, rel_x2)
    y_max = max(rel_y1, rel_y2)
    x_max = max(rel_x1, rel_x2)
    bbox = f"<loc{y_min}><loc{x_min}><loc{y_max}><loc{x_max}>"
    return bbox


def get_bbox_image(rgb_head_image:np.ndarray, 
                   bbox, target_height=224, target_width=224):
    """
    从图像中提取并调整 bbox 区域到目标尺寸
    
    Args:
        rgb_head_image: RGB 图像数组 (H, W, 3)
        bbox: 边界框 [x1, y1, x2, y2]
        target_height: 目标高度
        target_width: 目标宽度
        
    Returns:
        调整大小后的图像数组 (target_height, target_width, 3), dtype=uint8
    """
    # 确保图像是 float32 类型
    rgb_head_image = rgb_head_image.astype(np.float32)
    H, W, _ = rgb_head_image.shape

    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh)  # 使用最大边作为正方形边长
    cx, cy = x1 + bw / 2, y1 + bh / 2

    # 计算正方形 bbox
    new_x1 = int(np.floor(cx - side / 2))
    new_y1 = int(np.floor(cy - side / 2))
    new_x2 = int(np.ceil(cx + side / 2))
    new_y2 = int(np.ceil(cy + side / 2))

    # 计算需要的填充量（如果超出边界）
    pad_left = max(0, -new_x1)
    pad_top = max(0, -new_y1)
    pad_right = max(0, new_x2 - W)
    pad_bottom = max(0, new_y2 - H)

    # 使用 OpenCV 进行填充
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        img_padded = cv.copyMakeBorder(
            rgb_head_image,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv.BORDER_CONSTANT,
            value=0
        )
    else:
        img_padded = rgb_head_image

    # 更新裁剪坐标（考虑填充偏移）
    crop_x1 = new_x1 + pad_left
    crop_y1 = new_y1 + pad_top
    crop_x2 = new_x2 + pad_left
    crop_y2 = new_y2 + pad_top

    # 裁剪正方形区域
    crop = img_padded[crop_y1:crop_y2, crop_x1:crop_x2, :]
    
    # 使用 OpenCV 调整大小（双线性插值）
    crop_resized = cv.resize(
        crop, 
        (target_width, target_height), 
        interpolation=cv.INTER_LINEAR
    )
    
    # 转换为 uint8
    crop_resized = np.clip(crop_resized, 0, 255).astype(np.uint8)

    # 调试输出
    cv.imwrite("debug_condition_image.png",
               cv.cvtColor(crop_resized, cv.COLOR_RGB2BGR))
    
    return crop_resized.transpose(2, 0, 1)
