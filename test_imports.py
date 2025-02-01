# test_imports.py
try:
    print("Testing imports...")
    from deep_sort.preprocessing import non_max_suppression
    print("Successfully imported preprocessing")
    
    from deep_sort.detection import Detection
    print("Successfully imported detection")
    
    from deep_sort.tracker import Tracker
    print("Successfully imported tracker")
    
    print("All imports successful!")
except Exception as e:
    print(f"Error occurred: {str(e)}")