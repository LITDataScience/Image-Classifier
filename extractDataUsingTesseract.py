from PIL import Image
import pytesseract

info = {}

class EXTRACTINFO:

    def __init__(self, customer):

        self.customer = customer

        if self.customer == 1:
            self.path = './pic1.jpg'
        elif self.customer == 2:
            self.path = './pic2.jpg'
        else:
            self.path = './pic3.jpg'

        self.process(self.path)

    def process(self, path):
        response = self.extractFromImage(path)
        self.getInfo(response)

    def extractFromImage(self, path):
        resp = pytesseract.image_to_string(Image.open(path), lang='eng', nice=0)
        return resp


    def getInfo(self, response):

        PatientName, Age, DATE, Gender = ["" for _ in range(4)]

        for key in response.split():
            if 'name' in key.lower():
                sent = response.split()
                name = sent[sent.index(key) + 1]

            if 'age :' in key.lower():
                sent = response.split()
                Age = sent[sent.index(key) + 1]

            if 'dated' in key.lower():
                sent = response.split()
                DATE = sent[sent.index(key) + 1]

            if 'gender' in key.lower():
                sent = response.split()
                Gender = sent[sent.index(key) + 1]

        self.preparePatientData(PatientName, DATE, Age, Gender)


    def preparePatientData(self, name, DATE = 'DEFAULT', Age = 'DEFAULT', Gender = 'DEFAULT'):
        """
            Purpose: Add values to dictionary extracted from the image
        """
        info['NAME'] = name
        info['GENDER'] = Gender
        info['AGE'] = Age
        info['DATE'] = DATE
        print(info)

if __name__ == '__main__':
    customer = 2
    obj = EXTRACTINFO(customer)

