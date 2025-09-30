"""
FBRef scraper using Selenium to bypass anti-scraping measures.
This mimics human browser behavior to collect player pass data.

Requirements:
pip install selenium webdriver-manager pandas
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import random
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FBRefBrowserScraper:
    """Scraper that uses real browser to avoid detection."""

    def __init__(self, headless=False):
        """Initialize browser driver.

        Args:
            headless: Run without visible browser window
        """
        self.setup_driver(headless)
        self.output_dir = Path("data/raw/fbref_browser")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_driver(self, headless):
        """Setup Chrome driver with anti-detection measures."""
        chrome_options = Options()

        # Anti-detection measures
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # Additional options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")

        if headless:
            chrome_options.add_argument("--headless=new")  # New headless mode

        # Random user agent
        user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        chrome_options.add_argument(f'user-agent={random.choice(user_agents)}')

        # Create driver
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

        # Execute script to remove webdriver property
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    def human_delay(self, min_seconds=2, max_seconds=5):
        """Add random delay to mimic human behavior."""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)

    def scrape_match(self, match_url):
        """Scrape a single match with browser automation."""
        logger.info(f"Scraping: {match_url}")

        try:
            # Navigate to page
            self.driver.get(match_url)
            self.human_delay()

            # Wait for tables to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )

            # Scroll to trigger any lazy loading
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            self.human_delay(1, 2)

            # Find all tables
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            logger.info(f"Found {len(tables)} tables")

            # Extract player data from tables
            player_data = self.extract_player_data(tables)

            return player_data

        except Exception as e:
            logger.error(f"Error scraping match: {e}")
            return []

    def extract_player_data(self, tables):
        """Extract player passing data from page tables."""
        all_players = []

        # Tables of interest (based on FBRef structure)
        # Table 3-5: Home team stats
        # Table 10-12: Away team stats

        for i, table in enumerate(tables):
            try:
                # Check if this is a player stats table
                headers = table.find_elements(By.TAG_NAME, "th")
                header_text = [h.text for h in headers[:5]]

                # Look for player table indicators
                if any('Player' in h for h in header_text):
                    logger.info(f"Processing table {i} - likely player data")

                    # Extract rows
                    rows = table.find_elements(By.TAG_NAME, "tr")

                    for row in rows[1:]:  # Skip header row
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if not cells:
                            continue

                        # Extract player info (adjust indices based on actual structure)
                        player_dict = self.parse_player_row(cells, header_text)
                        if player_dict and player_dict.get('player_name'):
                            all_players.append(player_dict)

            except Exception as e:
                logger.debug(f"Error processing table {i}: {e}")
                continue

        logger.info(f"Extracted {len(all_players)} players")
        return all_players

    def parse_player_row(self, cells, headers):
        """Parse a single player row."""
        try:
            if len(cells) < 5:
                return None

            # Basic structure (adjust based on actual FBRef layout)
            player_dict = {
                'player_name': cells[0].text.strip(),
                'position': cells[3].text.strip() if len(cells) > 3 else '',
                'minutes_played': self.parse_minutes(cells[5].text if len(cells) > 5 else '0'),
            }

            # Look for passing columns (usually in separate table)
            for i, cell in enumerate(cells):
                cell_text = cell.text.strip()

                # Try to identify pass columns by position or content
                if i > 10 and i < 20:  # Typical range for passing stats
                    try:
                        if cell_text and cell_text.replace('.', '').isdigit():
                            if 'pass_att' not in player_dict:
                                player_dict['passes_attempted'] = float(cell_text)
                            elif 'pass_cmp' not in player_dict:
                                player_dict['passes_completed'] = float(cell_text)
                    except:
                        pass

            return player_dict

        except Exception as e:
            logger.debug(f"Error parsing row: {e}")
            return None

    def parse_minutes(self, min_str):
        """Parse minutes from string."""
        try:
            # Handle "90" or "90+3" format
            return int(min_str.split('+')[0])
        except:
            return 0

    def scrape_premier_league_recent(self, num_matches=5):
        """Scrape recent Premier League matches."""
        # Start with fixtures page
        fixtures_url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"

        logger.info("Getting Premier League fixtures...")
        self.driver.get(fixtures_url)
        self.human_delay()

        # Find match links
        match_links = []
        try:
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "scores"))
            )

            # Find all match report links
            links = self.driver.find_elements(By.PARTIAL_LINK_TEXT, "Match Report")

            for link in links[:num_matches]:
                href = link.get_attribute('href')
                if href and '/matches/' in href:
                    match_links.append(href)
                    logger.info(f"Found match: {href.split('/')[-1][:30]}...")

        except Exception as e:
            logger.error(f"Error getting match links: {e}")

        # Scrape each match
        all_player_data = []
        for i, match_url in enumerate(match_links):
            logger.info(f"\n[{i+1}/{len(match_links)}] Scraping match...")

            player_data = self.scrape_match(match_url)
            if player_data:
                for player in player_data:
                    player['match_url'] = match_url
                    player['match_id'] = i

                all_player_data.extend(player_data)

            # Random delay between matches
            if i < len(match_links) - 1:
                delay = random.uniform(5, 10)
                logger.info(f"Waiting {delay:.1f}s before next match...")
                time.sleep(delay)

        return all_player_data

    def save_data(self, player_data):
        """Save scraped data."""
        if not player_data:
            logger.warning("No data to save")
            return None

        df = pd.DataFrame(player_data)

        # Save to files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parquet_file = self.output_dir / f"fbref_browser_{timestamp}.parquet"
        csv_file = self.output_dir / f"fbref_browser_{timestamp}.csv"

        df.to_parquet(parquet_file, index=False)
        df.to_csv(csv_file, index=False)

        logger.info(f"Saved {len(df)} records to {parquet_file}")

        return df

    def close(self):
        """Close browser."""
        if hasattr(self, 'driver'):
            self.driver.quit()


def main():
    """Main function to run browser scraper."""
    print("\n" + "="*60)
    print("FBREF BROWSER SCRAPER")
    print("="*60)
    print("\nThis uses Selenium to automate a real browser.")
    print("It's slower but more likely to work.\n")

    # Check if user wants to proceed
    print("Options:")
    print("1. Run with visible browser (recommended for testing)")
    print("2. Run headless (background, no window)")
    print("3. Cancel")

    choice = input("\nChoice (1/2/3): ").strip()

    if choice == '3':
        print("Cancelled")
        return

    headless = (choice == '2')

    print("\nStarting browser scraper...")
    print("This will take several minutes due to human-like delays.\n")

    scraper = None
    try:
        # Initialize scraper
        scraper = FBRefBrowserScraper(headless=headless)

        # Scrape recent matches
        num_matches = 3  # Start small
        print(f"Scraping {num_matches} recent Premier League matches...")

        player_data = scraper.scrape_premier_league_recent(num_matches)

        if player_data:
            # Save data
            df = scraper.save_data(player_data)

            print(f"\n✓ Success! Scraped {len(df)} player records")

            if 'passes_attempted' in df.columns:
                print(f"Pass data available: {df['passes_attempted'].notna().sum()}/{len(df)} records")

            print("\nSample data:")
            print(df[['player_name', 'position', 'minutes_played']].head())

        else:
            print("\n✗ No data collected")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if scraper:
            scraper.close()
            print("\nBrowser closed")


if __name__ == "__main__":
    main()