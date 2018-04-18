import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


def send():
    from_addr = "E-MAIL HERE"
    to_addr = "E-MAIL HERE"

    msg = MIMEMultipart()

    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = "TREINAMENTO DA REDE CONCLUÍDO"

    # abre o arquivo results e coloca o conteúdo como corpo do e-mail
    with open('results.txt', 'r') as fp:
        body = fp.read()
    fp.close()

    msg.attach(MIMEText(body, 'plain'))

    # anexa o gráfico training loss
    with open('training_loss.png', 'rb') as fp:
        img = MIMEImage(fp.read())
    fp.close()
    img.add_header('Content-Disposition', 'attachment', filename='training_loss.png')
    msg.attach(img)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_addr, "YOUR PASSWORD HERE")
    text = msg.as_string()
    server.sendmail(from_addr, to_addr, text)
    server.quit()
