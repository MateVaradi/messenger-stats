# Imports
import sys
from data_cleaning import prepare_data
from aesthetics import set_aesthetics, set_colors
from plotting import MessengerReport


def run(conversation):
    # Load and clean data
    df = prepare_data(f"data/{conversation}/")

    # Aesthetic settings
    set_aesthetics()
    # Color settings
    members = sorted(df.sender_name.unique().tolist())
    color_dict = set_colors(members)

    reporter = MessengerReport(data=df, color_dict=color_dict, folder_name=conversation)
    reporter.pdf_report()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        conversation = sys.argv[1]
    else:
        raise ValueError("Add the name of the conversation as input")
    run(conversation)
