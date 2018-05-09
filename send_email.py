import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


def send(files):
    from_addr = "anderson.stan5@gmail.com"
    to_addr = "anderson.stan5@gmail.com"

    msg = MIMEMultipart()

    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = "Treinamento da Rede Neural concluído!"

    # abre o arquivo results e coloca o conteúdo como corpo do e-mail
    with open('results.txt', 'r') as fp:
        body = fp.read()

    msg.attach(MIMEText(body, 'plain'))

    for i in range(0, len(files)):
        with open(files[i], 'rb') as fp:
            file_attc = MIMEBase('application', "octet-stream")
            file_attc.set_payload(fp.read())
        encoders.encode_base64(file_attc)
        file_attc.add_header('Content-Disposition', 'attachment', filename=files[i])
        msg.attach(file_attc)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_addr, "cuoxaaiaaivqqsna")
    text = msg.as_string()
    server.sendmail(from_addr, to_addr, text)
    server.quit()
