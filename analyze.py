# Imports
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from data_cleaning import prepare_data
from aesthetics import set_fonts, set_aesthetics, set_colors
from plotting import MessengerReport


def run(conversation):
    # Load and clean data
    df = prepare_data("data/{conversation}")

    # Font settings
    set_fonts(fontsize=13)
    # Aesthetic settings
    set_aesthetics()
    # Color settings
    members = sorted(df.sender_name.unique().tolist())
    num_members = len(members)
    color_dict = set_colors(members)

    reporter = MessengerReport(data=df, color_dict=color_dict, folder_name=conversation)
    reporter.pdf_report()

    # TODO:
    # add message samples to pdf
    # finalize pdf report
    # figure out color_dict/palette

    # run for 2 other group chats

    # write blogpost


if __name__ == "__main__":
    run("data/technokratak3")
