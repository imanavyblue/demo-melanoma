from huggingface_hub import HfApi
import os

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if HF_API_TOKEN:
    api = HfApi()
    
    # ตรวจสอบให้แน่ใจว่าไฟล์มีอยู่ในระบบ
    model_path = "models/Inception_V3.h5"  # ตัวอย่าง: เส้นทางไปยังไฟล์โมเดล

    if os.path.exists(model_path):
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="Inception_V3.h5",
            repo_id="Suphawan/Melanoma-3",
            repo_type="model",
            token=HF_API_TOKEN
        )
        print("อัปโหลดโมเดลสำเร็จ")
    else:
        print(f"ไฟล์ {model_path} ไม่พบในระบบ")
else:
    print("ไม่พบ HF_API_TOKEN. กรุณาตรวจสอบการตั้งค่า API token.")
