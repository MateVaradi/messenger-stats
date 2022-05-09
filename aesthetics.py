""" Imports """
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib import ft2font
from matplotlib.font_manager import ttfFontProperty

""" Functions """


def set_emoji_font():
    fpath = "resources/TwitterColorEmoji-SVGinOT.ttf"
    font = ft2font.FT2Font(fpath)
    ttfFontProp = ttfFontProperty(font)
    fontprop = fm.FontProperties(
        family="sans-serif",
        fname=ttfFontProp.fname,
        size=20,
        stretch=ttfFontProp.stretch,
        style=ttfFontProp.style,
        variant=ttfFontProp.variant,
        weight=ttfFontProp.weight,
    )
    return fontprop


def set_aesthetics():
    sns.set_style("white")


def set_colors(members, colors=None):
    if colors is None:
        colors = ["#FA8879" "#96C8DF", "#BCE4CF", "#F9E031", "#C3D3D1"]
    color_dict = dict(zip(members, colors))
    return color_dict
