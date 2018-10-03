import re

file_path = open('./data/file1.txt', 'r')
MOBILE_REGEX = re.compile(r"^(?:(?:\+|0{0,2})91(\s*[\-]\s*)?|[0]?)?[6-9]\d{9}$")

EMAIL_REGEX = re.compile(r"^(([^<>()\[\]\.,;:\s@\"]+(\.[^<>()\[\]\.,;:\s@\"]+)*)|(\".+\"))@(([^<>()[\]\.,;:\s@\"]+\.)+[^<>()[\]\.,;:\s@\"]{2,})$")

DATE_IN_TEXT_REGEX = re.compile(r"\d{4}(?P<sep>[-/])\d{2}(?P=sep)\d{2}") # for qwqwd1994/09/09 type formats

DATE_REGEX = re.compile(r"\b(1[0-2]|0[1-9])/(3[01]|[12][0-9]|0[1-9])/[0-9]{4}\b") # for 09/09/1994 format

NAME_REGEX = re.compile(r"[A-Z]([a-z]+|\.)(?:\s+[A-Z]([a-z]+|\.))*(?:\s+[a-z][a-z\-]+){0,2}\s+[A-Z]([a-z]+|\.)")

LABNUMBER_REGEX = re.compile(r"[0-9]{7,9}") # as the lab number can be 7-9 digits long

GENDER_REGEX_MALE = re.compile(r"^[male]+")

for key in file_path:
    key = key.rstrip('\n')

    if re.match(MOBILE_REGEX, key):
        print('MOB FOUND: ', key)

    if re.match(EMAIL_REGEX, key):
        print('EMAIL FOUND: ', key)

    if re.match(DATE_IN_TEXT_REGEX, key):
        print('DATE IN TEXT FOUND: ', key)

    if re.match(DATE_REGEX, key):
        print('DATE FOUND: ', key)

    if re.match(NAME_REGEX, key):
        print('NAME FOUND: ', key)