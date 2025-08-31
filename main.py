from loguru import logger

from settings import settings
from utils import load_yaml_file
from recsys import Market360Recsys
from custom_types import ClientInput


CLIENTS = load_yaml_file(settings.clients_file_path)


class RecommendationPipeline:
    def __init__(self, recommendation_date: int) -> None:
        self.recommendation_date = recommendation_date

    def recommend(self):
        log_title = "[Recommendation]"

        logger.info(f"{log_title} Start")

        recommender = Market360Recsys(recommendation_date=self.recommendation_date)

        client_inputs = [ClientInput(**client) for client in CLIENTS if client['added_to_pipeline'] and client['schedule_region'] == settings.schedule_region]

        for client_input in client_inputs:
            logger.info(f"{log_title} Recommend for client {client_input.client_id}")

            try:
                recommendations = recommender.recommend(client=client_input)

                logger.info(f"{log_title} Generating report")
                report_html = recommender.generate_reports(client=client_input, recommendations=recommendations)
                logger.success(f"{log_title} Report generated")

                # prepare email template
                logger.info(f"{log_title} Preparing email template")
                email_template = recommender.send_email(client=client_input, recommendations=recommendations)
                logger.success(f"{log_title} Email template prepared")

            except Exception as e:
                logger.error(f"{log_title} Error: {e}")
                continue

        logger.success(f"{log_title} Recommendation DONE!")


if __name__ == "__main__":
    import optparse

    option_parser = optparse.OptonParser()
    option_parser.add_option("-d", "--date", dest="recommendation_date", default="20250822")

    options, args = option_parser.parse_args()

    pipeline = RecommendationPipeline(recommendation_date=options.recommendation_date)
    pipeline.recommend()