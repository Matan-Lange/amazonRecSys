import asyncio
import aiohttp
from aiofiles import open as aopen
import pandas as pd
from tqdm import tqdm
import os
from typing import List, Tuple
import sys
import time
from aiohttp import TCPConnector, ClientTimeout, ClientSession


class ImageDownloader:
    def __init__(self, max_concurrent: int = 10, timeout: int = 60, batch_size: int = 1000):
        self.max_concurrent = max_concurrent
        self.timeout = ClientTimeout(
            total=timeout,
            connect=30,
            sock_connect=30,
            sock_read=30
        )
        self.batch_size = batch_size
        self.success_count = 0
        self.failed_urls = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8'
        }

    async def download_image(self,
                           session: ClientSession,
                           args: Tuple[str, str, str, str],
                           retries: int = 3) -> bool:
        name, url, ext, output_folder = args
        filepath = f"{output_folder}/{name}.{ext}"

        # Skip if file already exists
        if os.path.exists(filepath):
            self.success_count += 1
            return True

        for attempt in range(retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        async with aopen(filepath, 'wb') as f:
                            await f.write(content)
                        self.success_count += 1
                        return True
                    elif response.status == 429:  # Too Many Requests
                        if attempt < retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                    self.failed_urls.append((url, f"Status code: {response.status}"))
                    return False
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(1)
                    continue
                self.failed_urls.append((url, str(e)))
                return False
        return False

    async def process_batch(self,
                          batch: List[Tuple[str, str, str, str]],
                          pbar: tqdm) -> None:
        connector = TCPConnector(
            limit=self.max_concurrent,
            force_close=False,
            enable_cleanup_closed=True,
            keepalive_timeout=60
        )

        async with ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers=self.headers
        ) as session:
            tasks = [self.download_image(session, args) for args in batch]
            for task in asyncio.as_completed(tasks):
                await task
                pbar.update(1)

    async def download_images_from_csv(self,
                                     csv_path: str,
                                     output_folder: str = 'images') -> None:
        os.makedirs(output_folder, exist_ok=True)
        images_urls_df = pd.read_csv(csv_path)

        download_args = [
            (name, url, ext, output_folder)
            for name, url, ext in zip(
                images_urls_df['parent_asin'],
                images_urls_df['large_image_url'],
                images_urls_df['image_format']
            )
        ]

        total_images = len(download_args)
        with tqdm(total=total_images, desc="Downloading images") as pbar:
            for i in range(0, total_images, self.batch_size):
                batch = download_args[i:i + self.batch_size]
                await self.process_batch(batch, pbar)
                if i + self.batch_size < total_images:
                    await asyncio.sleep(1)  # Delay between batches

        self._print_summary(total_images)

    def _print_summary(self, total: int) -> None:
        print(f"\nDownload Summary:")
        print(f"Successfully downloaded: {self.success_count}/{total}")
        print(f"Failed downloads: {len(self.failed_urls)}")

        # Save failed URLs to file
        if self.failed_urls:
            with open('failed_downloads.txt', 'w') as f:
                for url, error in self.failed_urls:
                    f.write(f"{url}: {error}\n")


def run_downloader(csv_path: str,
                  output_folder: str = 'images',
                  max_concurrent: int = 10,
                  timeout: int = 60,
                  batch_size: int = 1000):
    """Runner function with proper event loop handling"""
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        start_time = time.time()
        downloader = ImageDownloader(max_concurrent, timeout, batch_size)
        loop.run_until_complete(
            downloader.download_images_from_csv(csv_path, output_folder)
        )
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    finally:
        loop.close()


if __name__ == "__main__":
    csv_file = r'C:\Users\User\PycharmProjects\amazonRecSys\data\images_urls.csv'
    run_downloader(
        csv_file,
        'images',
        max_concurrent=10,
        timeout=60,
        batch_size=1000
    )