from detector import run_detection

print("=== AI Vision Assistant V3 ===\n")

print("Choose Camera Source:")
print("1 = Laptop Webcam")
print("2 = Phone IP Camera")

choice = input("Enter choice (1 or 2): ")

if choice == "2":
    ip = input("Enter phone IP (example 192.168.1.5): ")
    source = f"http://{ip}:8080/video"
else:
    source = 0

print("\nChoose Mode:")
print("1 = Detection Mode")
print("2 = Navigation Mode")

mode = input("Enter choice (1 or 2): ")

run_detection(source, mode)