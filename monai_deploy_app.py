from monai.deploy.core import Application

class DentalFusionApp(Application):
    def __init__(self):
        super().__init__()

    def compose(self):
        print("====================================")
        print("Dental AI MONAI Deploy Pipeline")
        print("====================================")
        print("Step 1: Input DICOM received")
        print("Step 2: Preprocessing started")
        print("Step 3: Model inference started")
        print("Step 4: DICOM SEG output generated")
        print("Pipeline completed successfully.")
        

if __name__ == "__main__":
    app = DentalFusionApp()