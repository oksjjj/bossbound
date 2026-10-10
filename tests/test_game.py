import unittest
import numpy as np
from game.core import Game,ACTIONS
from game.env import DungeonEnv
from agents.dqn import DQN
from tempfile import TemporaryDirectory

class Rules(unittest.TestCase):
    def test_equipment(self):
        for zone,owned,counter in [('hunt','weapon','weaponKills'),('ruins','armor','armorKills')]:
            g=Game();g.s['atk']=30
            for i in range(1,8):
                g.step(zone);g.step('attack');self.assertEqual(g.s[owned],i>=6);self.assertEqual(g.s[counter],min(i,6))
    def test_potions(self):
        g=Game();self.assertEqual(g.s['potions'],0);g.s['atk']=30
        for _ in range(12):g.step('forage');g.step('attack')
        self.assertEqual(g.s['potions'],4)
    def test_combat(self):
        g=Game();g.s.update(weapon=True,armor=True);g.step('boss');g.step('attack');self.assertEqual(g.s['hp'],26);self.assertEqual(g.s['enemy']['hp'],39)
        g.s['enemy']['phase']=2;g.step('guard');self.assertEqual(g.s['hp'],22);g.step('flee');self.assertEqual(g.s['hp'],18);g.step('boss');self.assertEqual(g.s['enemy']['hp'],48)
    def test_growth_and_invalid(self):
        g=Game();g.s.update(points=2,level=5);t=g.s['turn'];g.step('attackUp');g.step('defenseUp');self.assertEqual(g.s['turn'],t)
        before=g.observe()
        with self.assertRaises(ValueError):g.step('attack')
        self.assertEqual(g.s,before)
    def test_terminal(self):
        g=Game();g.s['turn']=89;g.step('train');self.assertTrue(g.s['done']);self.assertEqual(g.s['score'],-30)
        g=Game();g.s['atk']=48;g.s['turn']=88;g.step('boss');_,r,d,_=g.step('attack');self.assertTrue(d);self.assertEqual(r,100)
    def test_gym(self):
        from gymnasium.utils.env_checker import check_env
        check_env(DungeonEnv(),skip_render_check=True)
    def test_dqn(self):
        env=DungeonEnv();obs,info=env.reset();agent=DQN(len(obs),device='cpu');a=agent.act(obs,info['action_mask']);self.assertTrue(info['action_mask'][a]);n,r,d,_,ni=env.step(a)
        batch=(np.repeat(obs[None],8,0),np.full(8,a),np.zeros(8),np.repeat(n[None],8,0),np.ones(8,bool),np.zeros((8,len(ACTIONS)),bool))
        self.assertTrue(np.isfinite(agent.learn(batch)))
        with TemporaryDirectory() as p:
            agent.save(p+'/model.pt');loaded=DQN.load(p+'/model.pt','cpu');np.testing.assert_allclose(agent.q(obs),loaded.q(obs))
if __name__=='__main__':unittest.main()
