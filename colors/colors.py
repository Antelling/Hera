c = {
    "purple": '\033[95m',
    "blue": '\033[94m',
    "green": '\033[92m',
    "yellow": '\033[93m',
    "red": '\033[91m',
    "end": '\033[0m',
    "bold": '\033[1m',
    "underline": '\033[4m'
}


def red(text):
    print(c["red"] + str(text) + c["end"])


def blue(text):
    print(c["blue"] + str(text) + c["end"])


def green(text):
    print(c["green"] + str(text) + c["end"])


def yellow(text):
    print(c["yellow"] + str(text) + c["end"])


def purple(text):
    print(c["purple"] + str(text) + c["end"])

def white(text):
    print(str(text))

def bold(text):
    return c["bold"] + str(text) + c["end"]

def underline(text):
    return c["bold"] + str(text) + c["end"]


