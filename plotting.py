from tarfile import FIFOTYPE
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings

from aesthetics import set_emoji_font


class MessengerReport:
    def __init__(self, data, color_dict, folder_name, fontsize=13):
        self.data = data
        self.color_dict = color_dict
        self.folder_name = folder_name
        self.members = color_dict.keys()
        self.num_members = len(self.members)
        self.fontsize = fontsize
        self.reaction_columns = [
            c for c in self.data.columns if c.startswith("reaction_")
        ]
        self.emoji_font_prop = set_emoji_font()

    def plot_message_volume(self, return_fig=False):
        """Show number of messages and number of words sent per member"""
        # Prepare data
        df = self.data.copy()
        df["word_count"] = df["content"].apply(lambda x: x.count(" ") + 1)
        plot_data = (
            df.groupby("sender_name")
            .agg({"hour": "count", "word_count": "sum"})
            .rename(columns={"hour": "num_message", "word_count": "total_words"})
            .sort_values("sender_name")
            .reset_index()
        )
        plot_data["colors"] = plot_data.sender_name.map(self.color_dict)
        plot_data.sort_values("num_message", inplace=True, ascending=False)

        # Prepare plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Number of messages
        sns.barplot(
            data=plot_data,
            y="sender_name",
            x="num_message",
            orient="h",
            palette=plot_data.colors,
            saturation=1,
            ax=axes[0],
        )
        axes[0].set_xlabel("Number of messages")

        # Number of words
        plot_data.sort_values(
            "total_words", inplace=True, ascending=False, ignore_index=False
        )
        sns.barplot(
            data=plot_data,
            y="sender_name",
            x="total_words",
            orient="h",
            palette=plot_data.colors,
            saturation=1,
            ax=axes[1],
        )
        axes[1].set_xlabel("Number of words")

        # Aesthetics
        axes[0].set_ylabel("")
        axes[1].set_ylabel("")
        plt.suptitle("Messaging volume")
        plt.tight_layout()

        if return_fig:
            return fig
        else:
            fig.savefig(f"results/{self.folder_name}/messaging_volume.png", dpi=300)

    def plot_daily_use(self, return_fig=False):
        """Show daily use of group chat per member"""
        # Prepare data
        df = self.data.copy()
        total_days = (df.date_time.max() - df.date_time.min()).days
        plot_data = (
            df.groupby(["sender_name", "hour"])["content"]
            .count()
            .reset_index()
            .rename(columns={"content": "avg_num_msg"})
        )
        plot_data["avg_num_msg"] /= total_days
        plot_data["colors"] = plot_data.sender_name.map(self.color_dict)
        plot_data = plot_data.sort_values("sender_name")

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.lineplot(
            data=plot_data,
            x="hour",
            y="avg_num_msg",
            hue="sender_name",
            palette=list(self.color_dict.values()),
        )
        ax.set_ylabel("Avg. number of messages per day", size=13)
        ax.set_xlabel("Hour of the day", size=13)
        ax.set_title("Daily average messaging volume")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=13)
        plt.tight_layout()

        if return_fig:
            return fig
        else:
            fig.savefig(f"results/{self.folder_name}/daily_use.png", dpi=300)

    def plot_historic_use(self, return_fig=False):
        """Show historic messaging volume"""
        # Prepare data
        df = self.data.copy()
        plot_data = (
            df.groupby(["sender_name", "month"])["content"]
            .count()
            .reset_index()
            .rename(columns={"content": "num"})
        )

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.lineplot(
            data=plot_data,
            x="month",
            y="num",
            hue="sender_name",
            ax=ax,
            palette=list(self.color_dict.values()),
        )
        ax.set_title("Messaging volume over time", size=13)
        ax.set_xlabel("Date", size=13)
        x_labels = sorted([m[0:4] + "/" + m[4:] for m in plot_data.month.unique()])
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(labels=x_labels, rotation=45)
        ax.set_ylabel("Number of messages", size=13)
        plt.legend(title="Sender")
        plt.tight_layout()

        if return_fig:
            return fig
        else:
            fig.savefig(f"results/{self.folder_name}/volume_history.png", dpi=300)

    def sample_messages_with_emoji(
        self, emoji, sample_num=None, threshold=None, print_examples=True
    ):
        """Sample messages with a minimum number of a certain emoji"""
        df = self.data.copy()
        df["num_reaction"] = df[self.reaction_columns].apply(
            lambda x: list(x).count(emoji), axis=1
        )

        if threshold is None:
            threshold = len(self.members) - 1
        if sample_num is None:
            sample_num = 2 * len(self.members)

        # Print out or return messages
        texts = []
        df_to_sample = df.loc[
            ((df["num_reaction"] >= threshold) & (df["content"] != "")),
            ["sender_name", "content"] + self.reaction_columns,
        ]
        sample = df_to_sample.sample(sample_num)
        for _, row in sample.iterrows():
            reactions_received = row[self.reaction_columns].sum()
            text = (
                row["content"]
                + "\n  sent by "
                + row["sender_name"]
                + ", received "
                + reactions_received
                + "\n\n"
            )
            if print_examples:
                print(text)
            else:
                texts.append(text)

    def plot_emoji_received(self, emoji, threshold=1, return_fig=False):
        """Find how much each member received of a certain emoji"""
        # Prepare data
        df = self.data.copy()
        df["num_emoji_reaction"] = df[self.reaction_columns].apply(
            lambda x: list(x).count(emoji), axis=1
        )
        df_filtered = df.loc[
            df["num_emoji_reaction"] >= threshold, ["sender_name", "content"]
        ]
        plot_data = (
            df_filtered.groupby("sender_name")["content"]
            .count()[self.members]
            .reset_index()
        )
        plot_data["colors"] = plot_data.sender_name.map(self.color_dict)
        plot_data.sort_values("content", ascending=False, inplace=True)

        # Create plot
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            data=plot_data,
            y="sender_name",
            x="content",
            orient="h",
            palette=plot_data.colors,
            saturation=1,
        )
        ax.set_xlabel(
            f"Number of messages with at least {threshold} {emoji} reactions received",
            size=self.fontsize,
            fontproperties=self.emoji_font_prop,
        )
        ax.set_ylabel("")
        plt.tight_layout()

        if return_fig:
            return fig
        else:
            fig.savefig(
                f"results/{self.folder_name}/leaderboard_{threshold}_{emoji}.png",
                dpi=300,
            )

    def plot_reactions(self, return_fig=False):
        """Show most often sent and received reactions"""
        # Prepare data
        df = self.data.copy()

        # Create plots
        fig, axes = plt.subplots(
            self.num_members, 2, figsize=(12, 3 * self.num_members)
        )
        for i, member in enumerate(self.members):
            reactions_sent = (
                df.loc[df[f"reaction_{member}"] != "", f"reaction_{member}"]
                .value_counts()
                .reset_index()
                .rename(columns={f"reaction_{member}": "reactions"})
            )
            reactions_rec = (
                pd.Series(
                    list(
                        df.loc[df["sender_name"] == member, self.reaction_columns]
                        .sum(axis=1)
                        .sum(axis=0)
                    )
                )
                .value_counts()
                .reset_index()
                .rename(columns={0: "reactions"})
            )

            total_reaction_sent = reactions_sent["reactions"].sum()
            total_reaction_rec = reactions_rec["reactions"].sum()

            sns.barplot(
                data=reactions_rec.iloc[0:5],
                y="index",
                x="reactions",
                orient="h",
                color=self.color_dict[member],
                saturation=1,
                ax=axes[i, 0],
            )
            axes[i, 0].set_title(
                f"Most often received reactions\n(total reactions received: {total_reaction_rec})"
            )
            sns.barplot(
                data=reactions_sent.iloc[0:5],
                y="index",
                x="reactions",
                orient="h",
                color=self.color_dict[member],
                saturation=1,
                ax=axes[i, 1],
            )
            axes[i, 1].set_title(
                f"Most often sent reactions\n(total reactions sent: {total_reaction_sent})"
            )
            axes[i, 0].set_ylabel(member)
            axes[i, 1].set_ylabel("")
            axes[i, 0].set_xlabel("")
            axes[i, 1].set_xlabel("")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                axes[i, 0].set_yticklabels(
                    reactions_rec["index"][0:5],
                    size=self.fontsize * 1.5,
                    fontproperties=self.emoji_font_prop,
                )
                axes[i, 1].set_yticklabels(
                    reactions_sent["index"][0:5],
                    size=self.fontsize * 1.5,
                    fontproperties=self.emoji_font_prop,
                )

        plt.suptitle("Most common reactions")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()

        if return_fig:
            return fig
        else:
            fig.savefig(f"results/{self.folder_name}/reactions.png", dpi=300)

    def plot_reaction_network(self, emoji, return_fig=False):
        """Plot network of emoji reactions"""
        # Prepare data
        df = self.data.copy()
        # Filter dataframe to messages with selected emoji reaction
        emoji_df = df.loc[
            df[self.reaction_columns].apply(lambda x: emoji in list(x), axis=1),
            ["sender_name"] + self.reaction_columns,
        ]
        # Create matrix of reactions
        new_reaction_columns = []
        for col in self.reaction_columns:
            new_col = col.split("_")[1]
            new_reaction_columns.append(new_col)
            emoji_df[new_col] = emoji_df[col].apply(lambda x: 1 if x == emoji else 0)
            del emoji_df[col]
        reaction_matrix_df = (
            emoji_df.groupby("sender_name")[new_reaction_columns].sum().reset_index()
        )
        reaction_members = reaction_matrix_df.sender_name.tolist()
        reaction_matrix_df = reaction_matrix_df[
            ["sender_name"] + reaction_members
        ].set_index("sender_name")

        N = nx.MultiDiGraph()
        for member in reaction_members:
            m = member.replace(" ", "\n")
            N.add_node(m, color=self.color_dict[member])
        for sender in reaction_members:
            s = sender.replace(" ", "\n")
            for receiver in reaction_members:
                r = receiver.replace(" ", "\n")
                n = reaction_matrix_df[sender][receiver]
                N.add_edge(s, r, n)

        # Create graph plot
        fig, _ = plt.subplots(figsize=(12, 10))
        pos = nx.circular_layout(N)
        member_weights = [(sender, w) for sender, receiver, w in N.edges]
        reactions_sent_per_member = {
            sender.replace(" ", "\n"): w
            for sender, w in reaction_matrix_df.sum().to_dict().items()
        }
        widths = [
            w / reactions_sent_per_member[sender] * 4 for sender, w in member_weights
        ]
        nx.draw(
            N,
            with_labels=True,
            arrows=True,
            node_size=4500,
            font_size=self.fontsize,
            connectionstyle="arc3, rad = 0.1",
            width=widths,
            node_color=[self.color_dict[m.replace("\n", " ")] for m in N.nodes],
        )

        plt.suptitle(f"Network of {emoji} reactions")
        if return_fig:
            return fig
        else:
            fig.savefig(f"results/{self.folder_name}/network_of_{emoji}", dpi=300)

    def pdf_report(self, emojis=None):
        """
        Add all plots to a pdf report
        """
        # Select which emojis to analyze
        if emojis is None:
            emojis = ["😆"]

        plots = []
        # Create plots
        plots.append(self.plot_message_volume(return_fig=True))
        plots.append(self.plot_daily_use(return_fig=True))
        plots.append(self.plot_historic_use(return_fig=True))
        # self.sample_messages_with_emoji()
        plots.append(self.plot_reactions(return_fig=True))
        for emoji in emojis:
            plots.append(self.plot_emoji_received(emoji, threshold=1, return_fig=True))
            plots.append(
                self.plot_emoji_received(
                    emoji, threshold=self.num_members - 1, return_fig=True
                )
            )
            plots.append(self.plot_reaction_network(emoji, return_fig=True))

        # Create pdf
        report_pdf = PdfPages(
            f"results/{self.folder_name}/{self.folder_name}_report.pdf"
        )

        # Add title page
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.set_axis_off()
        plt.text(
            0.1,
            0.5,
            f"{self.folder_name}\nmessenger report",
            transform=fig.transFigure,
            size=self.fontsize * 2,
            fontweight="heavy",
        )
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        report_pdf.savefig(fig)

        # Add plots
        for plot in plots:
            report_pdf.savefig(plot, bbox_inches="tight", pad_inches=0.2)
        report_pdf.close()
