from process_input_keys import PressKey, ReleaseKey, W, A, S, D
import time

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

while True:
    PressKey(W)
    PressKey(A)
    time.sleep(3)