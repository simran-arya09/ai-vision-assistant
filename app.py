from detector import run_detection

print("Choose Camera Source:")
print("1 = Laptop Webcam")
print("2 = Phone IP Camera")

choice = input("Enter choice (1 or 2): ")

if choice == "2":
    source = "http://10.24.14.43:8080/video"
else:
    source = 0

run_detection(source)