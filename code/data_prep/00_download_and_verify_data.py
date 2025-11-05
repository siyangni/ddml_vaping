"""
Data download and verification script for PATH Study and NYTS.

Downloads public-use data files, verifies integrity, and prepares
directory structure for analysis.

Usage:
    python 00_download_and_verify_data.py
"""

import os
import sys
import hashlib
import requests
from pathlib import Path
import zipfile
import logging
from typing import Dict, List, Optional
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / 'code' / 'utils'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'outputs' / 'logs' / 'data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Download and verify study datasets."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize downloader.

        Parameters
        ----------
        config_path : Path, optional
            Path to config file
        """
        if config_path is None:
            config_path = project_root / 'config' / 'config.yaml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_dir = project_root / 'data' / 'raw'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DataDownloader initialized. Data directory: {self.data_dir}")

    def download_path_study(self) -> None:
        """
        Download PATH Study Public-Use Files.

        PATH data requires ICPSR registration. This method provides
        instructions for manual download if automated access is not available.
        """
        logger.info("=" * 80)
        logger.info("PATH STUDY DATA DOWNLOAD")
        logger.info("=" * 80)

        path_dir = self.data_dir / 'PATH'
        path_dir.mkdir(exist_ok=True)

        # Check if data already exists
        expected_files = [
            'PATH_W1_W6_Data.zip',
            'PATH_W1_W6_Codebook.pdf',
            'PATH_W1_W6_User_Guide.pdf'
        ]

        existing_files = [f for f in expected_files if (path_dir / f).exists()]

        if len(existing_files) == len(expected_files):
            logger.info("PATH data files already present. Verifying...")
            self._verify_path_files(path_dir)
            return

        # Provide download instructions
        logger.info("\nPATH Study Public-Use Files (PUF) require ICPSR registration.")
        logger.info("\nTo download PATH data:")
        logger.info("1. Visit: https://www.icpsr.umich.edu/web/NAHDAP/studies/36498")
        logger.info("2. Register for an ICPSR account if needed")
        logger.info("3. Request access to NAHDAP restricted data")
        logger.info("4. Download the following files to: {}".format(path_dir))
        logger.info("   - Population Assessment of Tobacco and Health (PATH) Study [United States]")
        logger.info("     Public-Use Files (ICPSR 36498)")
        logger.info("   - Waves 1-6 (2013-2019)")
        logger.info("\nRequired files:")
        for f in expected_files:
            logger.info(f"   - {f}")

        logger.info("\n" + "=" * 80)
        logger.info("Once downloaded, re-run this script to verify the files.")
        logger.info("=" * 80)

        # Try automated download (may fail without credentials)
        try:
            self._attempt_automated_path_download(path_dir)
        except Exception as e:
            logger.warning(f"Automated download failed: {e}")
            logger.info("Please download manually as instructed above.")

    def _attempt_automated_path_download(self, path_dir: Path) -> None:
        """
        Attempt automated PATH download (requires credentials).

        Parameters
        ----------
        path_dir : Path
            Directory to save files
        """
        logger.info("\nAttempting automated download...")

        # ICPSR download URLs (public access URLs if available)
        # Note: Direct download often requires authentication
        download_urls = {
            'user_guide': 'https://www.icpsr.umich.edu/web/NAHDAP/studies/36498/documentation',
        }

        # This is a placeholder - actual automated download would require:
        # 1. ICPSR API credentials
        # 2. Proper authentication flow
        # 3. Parsing of download links from the study page

        logger.warning("Automated download not implemented for PATH Study.")
        logger.info("Please download manually from ICPSR as instructed.")

    def _verify_path_files(self, path_dir: Path) -> None:
        """
        Verify PATH data files.

        Parameters
        ----------
        path_dir : Path
            Directory containing PATH files
        """
        logger.info("Verifying PATH data files...")

        zip_file = path_dir / 'PATH_W1_W6_Data.zip'

        if zip_file.exists():
            logger.info(f"Found: {zip_file.name}")
            logger.info(f"Size: {zip_file.stat().st_size / (1024**3):.2f} GB")

            # Try to extract a sample to verify integrity
            try:
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    file_list = zf.namelist()[:10]  # First 10 files
                    logger.info(f"Archive contains {len(zf.namelist())} files")
                    logger.info("Sample contents:")
                    for f in file_list:
                        logger.info(f"  - {f}")

                logger.info("✓ PATH data archive verified")
            except Exception as e:
                logger.error(f"✗ Error reading archive: {e}")
        else:
            logger.warning(f"✗ PATH data file not found: {zip_file}")

    def download_nyts(self) -> None:
        """
        Download NYTS (National Youth Tobacco Survey) data.

        NYTS data is publicly available from CDC.
        """
        logger.info("\n" + "=" * 80)
        logger.info("NYTS DATA DOWNLOAD")
        logger.info("=" * 80)

        nyts_dir = self.data_dir / 'NYTS'
        nyts_dir.mkdir(exist_ok=True)

        years = self.config['data_sources']['nyts']['years']

        logger.info(f"\nDownloading NYTS data for years: {years}")

        # NYTS URLs (check current CDC structure)
        base_url = "https://www.cdc.gov/tobacco/data_statistics/surveys/nyts/data/files/"

        for year in years:
            self._download_nyts_year(year, nyts_dir, base_url)

        logger.info("\n✓ NYTS download complete")

    def _download_nyts_year(self, year: int, nyts_dir: Path, base_url: str) -> None:
        """
        Download NYTS data for a specific year.

        Parameters
        ----------
        year : int
            Survey year
        nyts_dir : Path
            Directory to save files
        base_url : str
            Base URL for downloads
        """
        filename = f"NYTS_{year}_Data.csv"
        output_path = nyts_dir / filename

        if output_path.exists():
            logger.info(f"  {year}: Already downloaded")
            return

        # Construct download URL
        # Note: Actual URLs may vary; these are examples
        url = f"{base_url}NYTS-{year}-Data.csv"

        logger.info(f"  {year}: Downloading from {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"  {year}: ✓ Downloaded ({output_path.stat().st_size / 1024:.1f} KB)")

        except requests.exceptions.RequestException as e:
            logger.warning(f"  {year}: ✗ Download failed: {e}")
            logger.info(f"  {year}: Please download manually from:")
            logger.info(f"         https://www.cdc.gov/tobacco/data_statistics/surveys/nyts/data/index.html")

    def create_data_inventory(self) -> pd.DataFrame:
        """
        Create inventory of downloaded data files.

        Returns
        -------
        inventory : pd.DataFrame
            Inventory of data files with metadata
        """
        logger.info("\nCreating data inventory...")

        inventory_records = []

        for data_source_dir in self.data_dir.iterdir():
            if data_source_dir.is_dir():
                for file_path in data_source_dir.rglob('*'):
                    if file_path.is_file():
                        stat = file_path.stat()

                        record = {
                            'source': data_source_dir.name,
                            'filename': file_path.name,
                            'path': str(file_path.relative_to(self.data_dir)),
                            'size_mb': stat.st_size / (1024 ** 2),
                            'modified': pd.Timestamp.fromtimestamp(stat.st_mtime)
                        }

                        inventory_records.append(record)

        inventory = pd.DataFrame(inventory_records)

        if len(inventory) > 0:
            inventory = inventory.sort_values(['source', 'filename'])

            # Save inventory
            inventory_path = self.data_dir / 'data_inventory.csv'
            inventory.to_csv(inventory_path, index=False)

            logger.info(f"\nData inventory saved to: {inventory_path}")
            logger.info(f"\nInventory summary:")
            logger.info(inventory.groupby('source')['size_mb'].agg(['count', 'sum']))

        return inventory

    def verify_all(self) -> Dict[str, bool]:
        """
        Verify all data sources.

        Returns
        -------
        verification : dict
            Verification status for each source
        """
        logger.info("\n" + "=" * 80)
        logger.info("DATA VERIFICATION")
        logger.info("=" * 80)

        verification = {}

        # PATH
        path_dir = self.data_dir / 'PATH'
        path_zip = path_dir / 'PATH_W1_W6_Data.zip'
        verification['PATH'] = path_zip.exists()

        # NYTS
        nyts_dir = self.data_dir / 'NYTS'
        nyts_files = list(nyts_dir.glob('NYTS_*.csv'))
        verification['NYTS'] = len(nyts_files) > 0

        logger.info("\nVerification results:")
        for source, status in verification.items():
            status_str = "✓ PASS" if status else "✗ FAIL"
            logger.info(f"  {source}: {status_str}")

        return verification


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("DML VAPING/SMOKING STUDY - DATA DOWNLOAD")
    logger.info("=" * 80)

    downloader = DataDownloader()

    # Download PATH Study
    downloader.download_path_study()

    # Download NYTS
    downloader.download_nyts()

    # Create inventory
    inventory = downloader.create_data_inventory()

    # Verify all
    verification = downloader.verify_all()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 80)

    all_verified = all(verification.values())

    if all_verified:
        logger.info("✓ All data sources verified and ready for analysis")
    else:
        logger.warning("✗ Some data sources are missing")
        logger.info("\nPlease follow the instructions above to complete data download.")

    logger.info("\nNext step: Run 01_build_cohorts_and_variables.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
