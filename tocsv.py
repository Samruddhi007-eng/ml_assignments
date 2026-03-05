import pandas as pd

df = pd.read_csv("SMSSpamCollection",
                 sep="\t",
                 names=["label","message"])

df.to_csv("sms.csv", index=False)