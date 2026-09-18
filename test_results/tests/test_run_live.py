from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_live import publish


class LivePageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.state = dict(active=True, time=4., snapshots=61, threads=12,
            message='Synthetic test', runs=[
                dict(case='streaming_wave', level=2, cells=32, stage='Ready'),
                dict(case='streaming_wave', level=3, cells=64, stage='Waiting'),
                dict(case='streaming_wave', level=4, cells=128, stage='Waiting')])
        convergence = self.output/'plots'/'streaming_wave'/'convergence.png'
        convergence.parent.mkdir(parents=True)
        convergence.write_text('synthetic fixture')

    def add_profiles(self, level):
        profile = self.output/'plots'/'streaming_wave'/f'l{level}'/'profiles.png'
        profile.parent.mkdir(parents=True)
        profile.write_text('synthetic fixture')

    def test_shared_convergence_link_is_not_added_to_waiting_rows(self):
        self.add_profiles(2)
        publish(self.output, self.state)
        page = (self.output/'index.html').read_text()
        self.assertEqual(page.count('>Convergence</a>'), 2)  # table row and level-2 preview
        self.assertEqual(page.count('Waiting for output'), 2)

        self.add_profiles(3)
        self.state['runs'][1]['stage'] = 'Rendering movie'
        publish(self.output, self.state)
        page = (self.output/'index.html').read_text()
        self.assertEqual(page.count('>Convergence</a>'), 4)  # two rows and two previews
        self.assertEqual(page.count('Waiting for output'), 1)


if __name__ == '__main__':
    unittest.main()
