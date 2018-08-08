import pyautogui
import time


def main():
    counter = 1
    width, height = pyautogui.size()
    while True:
        time.sleep(5)
        pyautogui.click(width / 2, height / 2)
        print('Clicked %d times' % counter)
        counter += 1


if __name__ == '__main__':
    main()