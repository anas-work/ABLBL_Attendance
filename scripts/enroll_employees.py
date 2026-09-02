import sys
import yaml
from src.enrollment.enroll_service import EnrollmentService

def main():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    service = EnrollmentService(config=config)
    summary = service.run_enrollment()

    print("\n--- ENROLLMENT SUMMARY ---")
    print(f"Processed Photos: {summary['total_processed']}")
    print(f"Successful Enrollments: {summary['enrolled_success']}")
    print(f"Failed Enrollments: {summary['enrolled_failed']}")
    print(f"Total Vectors in FAISS Index: {summary['total_vectors_in_gallery']}")

if __name__ == "__main__":
    main()
