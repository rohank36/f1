
def convert_to_decimal(x):
    if x<0: 
        return 1 + 100/abs(x)
    else: 
        return 1 + x/100

def main(odds):
    for k,v in odds.items():
        odds[k] = 1/v
    return odds

def pretty_print_probs(probs):
    for k,v in probs.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    odds = {
        "K Antonelli YES to podium" : 9.00,
        "C Leclerc YES to podium": 9.00,
        "L Hamilton YES to podium": 7.00,
        "Piastri YES to podium": 1.28,
        "Max Verstappen YES to podium": 1.25,
        "G Russel YES to podum": 1.47,


    }
    probs = main(odds)
    pretty_print_probs(probs)
    