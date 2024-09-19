from huggingface_hub import HfApi
import os

# ตรวจสอบว่ามี token หรือไม่
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if HF_API_TOKEN:
    # สร้างอ็อบเจกต์ HfApi
    api = HfApi()

    # อัปโหลดไฟล์โมเดลไปยัง Hugging Face
    api.upload_file(
        path_or_fileobj="Inception_V3.h5",  # ไฟล์โมเดลในเครื่อง
        path_in_repo="Inception_V3.h5",  # ชื่อไฟล์ที่จะบันทึกใน Hugging Face repository
        repo_id="Suphawan/Melanoma-3",  # Repository ของคุณ
        repo_type="model",  # ประเภทของ repository
        token=HF_API_TOKEN  # ใช้ API token สำหรับการยืนยันตัวตน
    )
    print("อัปโหลดโมเดลเสร็จสมบูรณ์")
else:
    print("ไม่พบ HF_API_TOKEN. กรุณาตรวจสอบการตั้งค่า API token.")
