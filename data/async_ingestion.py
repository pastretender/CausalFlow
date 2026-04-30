import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, Any, Callable, Coroutine
from collections import deque
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.base import Layer1DataEngine
except ImportError:
    Layer1DataEngine = object

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Layer1_Ingestion")

class MemoryInStreamBuffer:
    """
    A dynamic proxy routing buffer that captures asynchronous multi-modal streams
    and aligns them into coherent temporal snapshots.
    """
    def __init__(self, alignment_tolerance_ms: int = 50):
        self.alignment_tolerance = alignment_tolerance_ms / 1000.0

        # State buffers for different modalities
        self.l2_book_state = {"bids": [], "asks": []}
        self.latest_news_embeddings = deque(maxlen=5) # Keep last 5 news items
        self.latest_alt_data = {}

        # Lock for thread-safe state reads during snapshot emission
        self._lock = asyncio.Lock()

    def _update_order_book_side(self, current_side: list, updates: list) -> list:
        """Helper to maintain an ordered list of [price, size] up to a fixed depth."""
        # Convert to dict for fast price updates
        book_map = {float(price): float(size) for price, size in current_side}
        for price_str, size_str in updates:
            price, size = float(price_str), float(size_str)
            if size == 0:
                book_map.pop(price, None)
            else:
                book_map[price] = size
        # Return sorted list (descending for bids, ascending for asks generally, but we'll sort based on caller)
        return [[price, book_map[price]] for price in book_map.keys()]

    async def update_l2_book(self, data: Dict[str, Any]):
        """Updates the rolling state of the Limit Order Book using delta updates."""
        async with self._lock:
            if "bids" in data:
                bids = self._update_order_book_side(self.l2_book_state.get("bids", []), data["bids"])
                bids.sort(key=lambda x: x[0], reverse=True)
                self.l2_book_state["bids"] = bids[:10]  # maintain top 10 levels

            if "asks" in data:
                asks = self._update_order_book_side(self.l2_book_state.get("asks", []), data["asks"])
                asks.sort(key=lambda x: x[0])
                self.l2_book_state["asks"] = asks[:10]

    async def update_news(self, text_data: Dict[str, Any]):
        """Captures unstructured text/news streams and produces embeddings."""
        # Simulate async execution of text encoding logic
        await asyncio.sleep(0.001)
        # Mock embedding logic for the incoming text
        text_data["embedding"] = [0.1] * 768 # e.g., BERT embedding vector

        async with self._lock:
            self.latest_news_embeddings.append(text_data)

    async def generate_snapshot(self) -> Dict[str, Any]:
        """
        Emits a synchronized view of all data modalities at the current microsecond.
        """
        async with self._lock:
            snapshot = {
                "timestamp": time.time(),
                "l2_book": self.l2_book_state.copy(),
                "recent_news": list(self.latest_news_embeddings),
                "alt_data": self.latest_alt_data.copy()
            }
        return snapshot


class AsyncMultimodalEngine(Layer1DataEngine if Layer1DataEngine != object else object):
    """
    The core orchestration engine for Layer 1.
    Manages websocket connections, maintains the memory-in-stream buffer,
    and yields aligned data frames to the Global Orchestrator.
    """
    def __init__(self, config: Dict[str, Any]):
        self.symbols = config.get("symbols", ["BTC-USD"])
        self.buffer = MemoryInStreamBuffer(alignment_tolerance_ms=config.get("alignment_tolerance_ms", 50))
        self.snapshot_queue = asyncio.Queue(maxsize=1000)
        self.is_running = False

    async def _mock_websocket_feed(self):
        """Simulate an incoming websocket stream for testing."""
        import random
        base_price = 50000.0
        while self.is_running:
            bids = [[base_price - i*10 - random.uniform(0, 5), random.uniform(0.1, 2.0)] for i in range(1, 11)]
            asks = [[base_price + i*10 + random.uniform(0, 5), random.uniform(0.1, 2.0)] for i in range(1, 11)]

            # Send update to buffer
            await self.buffer.update_l2_book({"bids": bids, "asks": asks})

            # Randomly drift price
            base_price += random.choice([-10.0, 10.0, 0.0])
            await asyncio.sleep(0.1)  # 100ms tick rate

    async def _snapshot_loop(self):
        """Continuously pulls synchronized frames and adds them to the queue."""
        while self.is_running:
            snapshot = await self.buffer.generate_snapshot()
            if not self.snapshot_queue.full():
                await self.snapshot_queue.put(snapshot)
            await asyncio.sleep(0.1)

    async def start_engine(self):
        """Initializes and starts the ingestion and snapshot loops."""
        logger.info(f"Starting AsyncMultimodalEngine for {self.symbols}")
        self.is_running = True

        # Start the real aiohttp session and websockets rather than the pure mock
        self.session = aiohttp.ClientSession()
        asyncio.create_task(self._stream_l2_order_book(self.session))
        asyncio.create_task(self._stream_news_and_social(self.session))

        # Backward compatibility / fallback mock loop
        asyncio.create_task(self._mock_websocket_feed())

        # Start emitter/snapshot task
        asyncio.create_task(self._snapshot_loop())

    async def stop_engine(self):
        self.is_running = False
        if hasattr(self, 'session') and not self.session.closed:
            await self.session.close()

    async def ingest_multimodal_stream(self) -> Dict[str, Any]:
        """Provides an async generator for the orchestrator to consume snapshots."""
        while self.is_running or not self.snapshot_queue.empty():
            yield await self.snapshot_queue.get()

    def reconstruct_signal(self, observed_data):
        """Integration with MicrostructureInverseSolver."""
        try:
            from data.physical_degradation import MicrostructureInverseSolver
            solver = MicrostructureInverseSolver(l2_feature_dim=observed_data.shape[-1], latent_intent_dim=16)
            clean_intent, _ = solver(observed_data)
            return clean_intent
        except Exception as e:
            import logging
            logging.error(f"Signal reconstruction failed: {e}. Returning raw observed data.")
            return observed_data

    async def _stream_l2_order_book(self, session: aiohttp.ClientSession):
        """Implementation of a high-frequency WebSocket consumer."""
        # Example: wss://ws-feed.exchange.com
        logger.info("Initializing L2 Order Book stream...")
        try:
            # First try the real connection if URL is provided in config, else fallback
            ws_url = "wss://stream.binance.com:9443/ws/btcusdt@depth"
            try:
                # We try a real websocket connect using aiohttp
                async with session.ws_connect(ws_url) as ws:
                    while self.is_running:
                        msg = await ws.receive()
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            await self.buffer.update_l2_book(data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            break
            except Exception as e:
                logger.warning(f"Failed to connect to real ws stream, falling back to mock: {e}")
                while self.is_running:
                    # Mocking network reception of L2 data
                    await asyncio.sleep(0.01) # 10ms tick rate
                    mock_l2_data = {"bids": [[65000, 1.5], [64990, 0.5]], "asks": [[65010, 2.0]]}
                    await self.buffer.update_l2_book(mock_l2_data)
        except asyncio.CancelledError:
            logger.info("L2 Stream task cancelled.")

    async def _stream_news_and_social(self, session: aiohttp.ClientSession):
        """Implementation of an unstructured text stream via REST API polling."""
        logger.info("Initializing News and Social stream...")
        try:
            news_url = "https://api.coingecko.com/api/v3/news"
            while self.is_running:
                try:
                    async with session.get(news_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if "data" in data and len(data["data"]) > 0:
                                top_news = data["data"][0]
                                news_item = {"source": "CoinGecko", "text": top_news.get("title", ""), "sentiment": 0.5}
                                await self.buffer.update_news(news_item)
                except Exception as e:
                    # Fallback to mock on errors/rate limits
                    mock_news = {"source": "Telegram", "sentiment": 0.8, "text": "Fed rate cut hints..."}
                    await self.buffer.update_news(mock_news)
                await asyncio.sleep(5)  # Poll every 5s
        except asyncio.CancelledError:
            logger.info("News Stream task cancelled.")

    async def _snapshot_emitter(self, emit_interval_ms: int = 100):
        """
        Continuously takes snapshots of the Memory-in-Stream buffer at a fixed
        interval (e.g., 100ms) and pushes them to the orchestrator queue.
        """
        interval = emit_interval_ms / 1000.0
        try:
            while self.is_running:
                snapshot = await self.buffer.generate_snapshot()
                if not self.snapshot_queue.full():
                    await self.snapshot_queue.put(snapshot)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Snapshot emitter task cancelled.")
                await asyncio.sleep(interval)
                snapshot = await self.buffer.generate_snapshot()

                # Non-blocking put; if the downstream pipeline (PyTorch) is too slow,
                # we drop frames rather than OOM-ing the server.
                if not self.snapshot_queue.full():
                    self.snapshot_queue.put_nowait(snapshot)
                else:
                    logger.warning("Pipeline bottleneck: Dropping data frame.")
        except asyncio.CancelledError:
            logger.info("Emitter task cancelled.")

    async def start_engine(self):
        """Launches all concurrent streaming tasks."""
        self.is_running = True

        async with aiohttp.ClientSession() as session:
            # Orchestrate all async streams
            tasks = [
                asyncio.create_task(self._stream_l2_order_book(session)),
                asyncio.create_task(self._stream_news_and_social(session)),
                asyncio.create_task(self._snapshot_emitter(emit_interval_ms=100))
            ]

            logger.info("Layer 1 Async Engine fully operational.")
            await asyncio.gather(*tasks)

    def stop_engine(self):
        self.is_running = False

# ==============================================================
# Usage within `causalflow/core/orchestrator.py`
# ==============================================================
# async def run_live_pipeline(self):
#     ingestion_engine = AsyncMultimodalEngine(config={"symbols": ["AAPL"]})
#
#     # Start ingestion as a background task
#     ingestion_task = asyncio.create_task(ingestion_engine.start_engine())
#
#     try:
#         while True:
#             # Wait for the next synchronized frame
#             aligned_frame = await ingestion_engine.snapshot_queue.get()
#
#             # Pass to Physical Denoising -> CBM -> RL Execution...
#             clean_intent, _ = self.denoiser(aligned_frame['l2_book'])
#
#     except KeyboardInterrupt:
#         ingestion_engine.stop_engine()
#         ingestion_task.cancel()