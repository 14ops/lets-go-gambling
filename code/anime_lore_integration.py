import random
import json
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class AnimeCharacter:
    name: str
    anime: str
    strategy_type: str
    rarity: str  # "Common", "Rare", "Epic", "Legendary", "Mythic"
    power_level: int
    special_ability: str
    win_rate: float
    description: str
    portrait_svg: str = ""

@dataclass
class GachaConfig:
    pull_cost: int = 100
    guaranteed_rare_after: int = 10
    rarity_rates: Dict[str, float] = field(default_factory=lambda: {
        "Common": 0.60,
        "Rare": 0.25,
        "Epic": 0.10,
        "Legendary": 0.04,
        "Mythic": 0.01
    })

class AnimeLoreIntegration:
    """
    Enhances the framework with anime character lore, SVG portraits, and a gacha system.
    
    Like assembling the ultimate anime dream team where each character brings
    their unique abilities to the strategic battlefield!
    """
    
    def __init__(self):
        self.characters = self._initialize_characters()
        self.gacha_config = GachaConfig()
        self.user_collection = {}
        self.user_currency = 1000  # Starting currency
        self.pity_counter = 0  # For guaranteed rare pulls
        
    def _initialize_characters(self) -> Dict[str, AnimeCharacter]:
        """Initialize the anime character database."""
        characters = {
            "takeshi": AnimeCharacter(
                name="Takeshi Kovacs",
                anime="Altered Carbon",
                strategy_type="Aggressive Berserker",
                rarity="Rare",
                power_level=75,
                special_ability="Envoy Intuition - Rapid adaptation to combat scenarios",
                win_rate=0.52,
                description="A former Envoy with enhanced combat instincts and aggressive tactics. Takeshi charges into battle with reckless abandon, relying on raw power and intuition.",
                portrait_svg=self._generate_takeshi_svg()
            ),
            "lelouch": AnimeCharacter(
                name="Lelouch vi Britannia",
                anime="Code Geass",
                strategy_type="Strategic Mastermind",
                rarity="Legendary",
                power_level=95,
                special_ability="Geass - Command absolute obedience and strategic foresight",
                win_rate=0.64,
                description="The exiled prince with the power of Geass. Lelouch's brilliant tactical mind and supernatural command ability make him a formidable strategic opponent.",
                portrait_svg=self._generate_lelouch_svg()
            ),
            "kazuya": AnimeCharacter(
                name="Kazuya Kinoshita",
                anime="Rent-a-Girlfriend",
                strategy_type="Conservative Survivor",
                rarity="Common",
                power_level=45,
                special_ability="Rental Wisdom - Cautious decision-making and risk aversion",
                win_rate=0.78,
                description="A college student who learned caution through expensive rental relationships. Kazuya's conservative approach prioritizes survival over glory.",
                portrait_svg=self._generate_kazuya_svg()
            ),
            "senku": AnimeCharacter(
                name="Senku Ishigami",
                anime="Dr. Stone",
                strategy_type="Analytical Scientist",
                rarity="Epic",
                power_level=90,
                special_ability="Science Kingdom - Data-driven analysis and logical deduction",
                win_rate=0.72,
                description="A genius scientist who approaches every problem with logic and data. Senku's analytical mind turns complex situations into solvable equations.",
                portrait_svg=self._generate_senku_svg()
            ),
            "rintaro_okabe": AnimeCharacter(
                name="Rintaro Okabe",
                anime="Steins;Gate",
                strategy_type="Mad Scientist",
                rarity="Mythic",
                power_level=100,
                special_ability="Reading Steiner - Ability to perceive alternate timelines and outcomes",
                win_rate=0.85,
                description="The self-proclaimed mad scientist with the ability to retain memories across worldlines. Okabe's unique perspective allows him to see patterns others cannot.",
                portrait_svg=self._generate_okabe_svg()
            ),
            "kurisu": AnimeCharacter(
                name="Kurisu Makise",
                anime="Steins;Gate",
                strategy_type="Analytical Genius",
                rarity="Legendary",
                power_level=92,
                special_ability="Genius Intellect - Superior analytical and deductive reasoning",
                win_rate=0.76,
                description="A brilliant neuroscientist with exceptional analytical abilities. Kurisu's scientific approach and logical thinking make her a formidable strategist.",
                portrait_svg=self._generate_kurisu_svg()
            ),
            "light": AnimeCharacter(
                name="Light Yagami",
                anime="Death Note",
                strategy_type="Psychological Manipulator",
                rarity="Legendary",
                power_level=88,
                special_ability="Kira's Judgment - Psychological manipulation and strategic planning",
                win_rate=0.82,
                description="A brilliant student with a god complex. Light's ability to manipulate others and plan several steps ahead makes him a dangerous opponent.",
                portrait_svg=self._generate_light_svg()
            ),
            "edward": AnimeCharacter(
                name="Edward Elric",
                anime="Fullmetal Alchemist",
                strategy_type="Adaptive Alchemist",
                rarity="Epic",
                power_level=85,
                special_ability="Equivalent Exchange - Balanced risk-reward calculations",
                win_rate=0.70,
                description="The Fullmetal Alchemist who understands the principle of equivalent exchange. Edward's balanced approach weighs risks and rewards carefully.",
                portrait_svg=self._generate_edward_svg()
            )
        }
        return characters
    
    def _generate_takeshi_svg(self) -> str:
        """Generate SVG portrait for Takeshi."""
        return '''<svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="200" height="200" fill="#1a1a2e"/>
            <!-- Face -->
            <circle cx="100" cy="90" r="40" fill="#d4a574" stroke="#b8956a" stroke-width="2"/>
            <!-- Hair -->
            <path d="M60 70 Q100 40 140 70 Q130 50 100 45 Q70 50 60 70" fill="#2c1810"/>
            <!-- Eyes -->
            <circle cx="85" cy="85" r="3" fill="#000"/>
            <circle cx="115" cy="85" r="3" fill="#000"/>
            <!-- Scar -->
            <path d="M75 75 L90 80" stroke="#8b0000" stroke-width="2" fill="none"/>
            <!-- Mouth -->
            <path d="M90 105 Q100 110 110 105" stroke="#000" stroke-width="2" fill="none"/>
            <!-- Body -->
            <rect x="70" y="130" width="60" height="70" fill="#333" rx="5"/>
            <!-- Text -->
            <text x="100" y="180" text-anchor="middle" fill="#ff6b6b" font-family="Arial" font-size="12" font-weight="bold">TAKESHI</text>
            <text x="100" y="195" text-anchor="middle" fill="#ffd93d" font-family="Arial" font-size="8">BERSERKER</text>
        </svg>'''
    
    def _generate_lelouch_svg(self) -> str:
        """Generate SVG portrait for Lelouch."""
        return '''<svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="200" height="200" fill="#4a0e4e"/>
            <!-- Face -->
            <circle cx="100" cy="90" r="40" fill="#f5deb3" stroke="#e6d3a3" stroke-width="2"/>
            <!-- Hair -->
            <path d="M60 65 Q80 35 100 40 Q120 35 140 65 Q130 45 100 35 Q70 45 60 65" fill="#1a1a1a"/>
            <!-- Eyes (Geass) -->
            <circle cx="85" cy="85" r="4" fill="#ff0000"/>
            <circle cx="115" cy="85" r="4" fill="#ff0000"/>
            <circle cx="85" cy="85" r="2" fill="#fff"/>
            <circle cx="115" cy="85" r="2" fill="#fff"/>
            <!-- Mouth -->
            <path d="M90 105 Q100 108 110 105" stroke="#000" stroke-width="2" fill="none"/>
            <!-- Cape -->
            <path d="M50 130 Q100 125 150 130 L150 200 L50 200 Z" fill="#800080"/>
            <!-- Text -->
            <text x="100" y="180" text-anchor="middle" fill="#ff69b4" font-family="Arial" font-size="12" font-weight="bold">LELOUCH</text>
            <text x="100" y="195" text-anchor="middle" fill="#ffd700" font-family="Arial" font-size="8">MASTERMIND</text>
        </svg>'''
    
    def _generate_kazuya_svg(self) -> str:
        """Generate SVG portrait for Kazuya."""
        return '''<svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="200" height="200" fill="#87ceeb"/>
            <!-- Face -->
            <circle cx="100" cy="90" r="40" fill="#fdbcb4" stroke="#f5a9a9" stroke-width="2"/>
            <!-- Hair -->
            <path d="M65 70 Q100 50 135 70 Q125 55 100 50 Q75 55 65 70" fill="#8b4513"/>
            <!-- Eyes -->
            <circle cx="85" cy="85" r="3" fill="#000"/>
            <circle cx="115" cy="85" r="3" fill="#000"/>
            <!-- Blush -->
            <circle cx="70" cy="95" r="5" fill="#ff69b4" opacity="0.5"/>
            <circle cx="130" cy="95" r="5" fill="#ff69b4" opacity="0.5"/>
            <!-- Mouth -->
            <path d="M90 105 Q100 100 110 105" stroke="#000" stroke-width="2" fill="none"/>
            <!-- Shirt -->
            <rect x="70" y="130" width="60" height="70" fill="#4169e1" rx="5"/>
            <!-- Text -->
            <text x="100" y="180" text-anchor="middle" fill="#32cd32" font-family="Arial" font-size="12" font-weight="bold">KAZUYA</text>
            <text x="100" y="195" text-anchor="middle" fill="#ffd700" font-family="Arial" font-size="8">SURVIVOR</text>
        </svg>'''
    
    def _generate_senku_svg(self) -> str:
        """Generate SVG portrait for Senku."""
        return '''<svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="200" height="200" fill="#2e8b57"/>
            <!-- Face -->
            <circle cx="100" cy="90" r="40" fill="#fdbcb4" stroke="#f5a9a9" stroke-width="2"/>
            <!-- Hair -->
            <path d="M60 65 Q70 40 85 45 Q100 35 115 45 Q130 40 140 65" fill="#f0f8ff"/>
            <path d="M85 45 Q100 30 115 45" fill="#e6e6fa"/>
            <!-- Eyes -->
            <circle cx="85" cy="85" r="3" fill="#000"/>
            <circle cx="115" cy="85" r="3" fill="#000"/>
            <!-- Stone marks -->
            <path d="M75 80 L80 85" stroke="#32cd32" stroke-width="2"/>
            <path d="M120 85 L125 80" stroke="#32cd32" stroke-width="2"/>
            <!-- Mouth (smirk) -->
            <path d="M90 105 Q105 110 110 105" stroke="#000" stroke-width="2" fill="none"/>
            <!-- Lab coat -->
            <rect x="70" y="130" width="60" height="70" fill="#ffffff" rx="5"/>
            <!-- Text -->
            <text x="100" y="180" text-anchor="middle" fill="#00ff00" font-family="Arial" font-size="12" font-weight="bold">SENKU</text>
            <text x="100" y="195" text-anchor="middle" fill="#ffd700" font-family="Arial" font-size="8">SCIENTIST</text>
        </svg>'''
    
    def _generate_okabe_svg(self) -> str:
        """Generate SVG portrait for Okabe."""
        return '''<svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="200" height="200" fill="#2f1b69"/>
            <!-- Face -->
            <circle cx="100" cy="90" r="40" fill="#fdbcb4" stroke="#f5a9a9" stroke-width="2"/>
            <!-- Hair -->
            <path d="M60 65 Q80 30 100 35 Q120 30 140 65 Q130 40 100 25 Q70 40 60 65" fill="#8b4513"/>
            <!-- Eyes (Reading Steiner) -->
            <circle cx="85" cy="85" r="4" fill="#00ffff"/>
            <circle cx="115" cy="85" r="4" fill="#00ffff"/>
            <circle cx="85" cy="85" r="2" fill="#fff"/>
            <circle cx="115" cy="85" r="2" fill="#fff"/>
            <!-- Mouth (maniacal grin) -->
            <path d="M85 105 Q100 115 115 105" stroke="#000" stroke-width="2" fill="none"/>
            <!-- Lab coat -->
            <rect x="70" y="130" width="60" height="70" fill="#ffffff" rx="5"/>
            <!-- Tie -->
            <rect x="95" y="130" width="10" height="40" fill="#ff0000"/>
            <!-- Text -->
            <text x="100" y="180" text-anchor="middle" fill="#ff1493" font-family="Arial" font-size="12" font-weight="bold">OKABE</text>
            <text x="100" y="195" text-anchor="middle" fill="#ffd700" font-family="Arial" font-size="8">MAD SCIENTIST</text>
        </svg>'''
    
    def _generate_kurisu_svg(self) -> str:
        """Generate SVG portrait for Kurisu."""
        return '''<svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="200" height="200" fill="#8b0000"/>
            <!-- Face -->
            <circle cx="100" cy="90" r="40" fill="#fdbcb4" stroke="#f5a9a9" stroke-width="2"/>
            <!-- Hair -->
            <path d="M60 70 Q100 40 140 70 Q130 50 100 45 Q70 50 60 70" fill="#dc143c"/>
            <!-- Eyes -->
            <circle cx="85" cy="85" r="3" fill="#4169e1"/>
            <circle cx="115" cy="85" r="3" fill="#4169e1"/>
            <!-- Mouth -->
            <path d="M90 105 Q100 108 110 105" stroke="#000" stroke-width="2" fill="none"/>
            <!-- Jacket -->
            <rect x="70" y="130" width="60" height="70" fill="#800080" rx="5"/>
            <!-- Text -->
            <text x="100" y="180" text-anchor="middle" fill="#ff69b4" font-family="Arial" font-size="12" font-weight="bold">KURISU</text>
            <text x="100" y="195" text-anchor="middle" fill="#ffd700" font-family="Arial" font-size="8">GENIUS</text>
        </svg>'''
    
    def _generate_light_svg(self) -> str:
        """Generate SVG portrait for Light."""
        return '''<svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="200" height="200" fill="#1a1a1a"/>
            <!-- Face -->
            <circle cx="100" cy="90" r="40" fill="#fdbcb4" stroke="#f5a9a9" stroke-width="2"/>
            <!-- Hair -->
            <path d="M60 65 Q80 35 100 40 Q120 35 140 65 Q130 45 100 30 Q70 45 60 65" fill="#daa520"/>
            <!-- Eyes (Shinigami eyes) -->
            <circle cx="85" cy="85" r="4" fill="#ff0000"/>
            <circle cx="115" cy="85" r="4" fill="#ff0000"/>
            <circle cx="85" cy="85" r="2" fill="#000"/>
            <circle cx="115" cy="85" r="2" fill="#000"/>
            <!-- Mouth (sinister smile) -->
            <path d="M85 105 Q100 112 115 105" stroke="#000" stroke-width="2" fill="none"/>
            <!-- Shirt -->
            <rect x="70" y="130" width="60" height="70" fill="#ffffff" rx="5"/>
            <!-- Text -->
            <text x="100" y="180" text-anchor="middle" fill="#ff0000" font-family="Arial" font-size="12" font-weight="bold">LIGHT</text>
            <text x="100" y="195" text-anchor="middle" fill="#ffd700" font-family="Arial" font-size="8">KIRA</text>
        </svg>'''
    
    def _generate_edward_svg(self) -> str:
        """Generate SVG portrait for Edward."""
        return '''<svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="200" height="200" fill="#8b4513"/>
            <!-- Face -->
            <circle cx="100" cy="90" r="40" fill="#fdbcb4" stroke="#f5a9a9" stroke-width="2"/>
            <!-- Hair -->
            <path d="M60 70 Q100 40 140 70 Q130 50 100 45 Q70 50 60 70" fill="#ffd700"/>
            <!-- Braid -->
            <path d="M140 70 Q150 80 145 90 Q155 100 150 110" stroke="#ffd700" stroke-width="4" fill="none"/>
            <!-- Eyes -->
            <circle cx="85" cy="85" r="3" fill="#ffd700"/>
            <circle cx="115" cy="85" r="3" fill="#ffd700"/>
            <!-- Mouth -->
            <path d="M90 105 Q100 110 110 105" stroke="#000" stroke-width="2" fill="none"/>
            <!-- Red coat -->
            <rect x="70" y="130" width="60" height="70" fill="#dc143c" rx="5"/>
            <!-- Automail arm -->
            <rect x="130" y="140" width="15" height="50" fill="#c0c0c0" rx="2"/>
            <!-- Text -->
            <text x="100" y="180" text-anchor="middle" fill="#ffd700" font-family="Arial" font-size="12" font-weight="bold">EDWARD</text>
            <text x="100" y="195" text-anchor="middle" fill="#ffd700" font-family="Arial" font-size="8">ALCHEMIST</text>
        </svg>'''
    
    def perform_gacha_pull(self, num_pulls: int = 1) -> List[AnimeCharacter]:
        """Perform gacha pulls to obtain characters."""
        if self.user_currency < self.gacha_config.pull_cost * num_pulls:
            raise ValueError("Insufficient currency for gacha pull!")
        
        self.user_currency -= self.gacha_config.pull_cost * num_pulls
        pulled_characters = []
        
        for _ in range(num_pulls):
            self.pity_counter += 1
            
            # Determine rarity
            if self.pity_counter >= self.gacha_config.guaranteed_rare_after:
                # Guaranteed rare or higher
                rarity_pool = ["Rare", "Epic", "Legendary", "Mythic"]
                weights = [0.70, 0.20, 0.08, 0.02]
                rarity = np.random.choice(rarity_pool, p=weights)
                self.pity_counter = 0
            else:
                # Normal pull
                rarities = list(self.gacha_config.rarity_rates.keys())
                weights = list(self.gacha_config.rarity_rates.values())
                rarity = np.random.choice(rarities, p=weights)
                
                if rarity in ["Rare", "Epic", "Legendary", "Mythic"]:
                    self.pity_counter = 0
            
            # Select character of determined rarity
            available_chars = [char for char in self.characters.values() if char.rarity == rarity]
            if available_chars:
                character = random.choice(available_chars)
                pulled_characters.append(character)
                
                # Add to collection
                if character.name in self.user_collection:
                    self.user_collection[character.name] += 1
                else:
                    self.user_collection[character.name] = 1
        
        return pulled_characters
    
    def display_character_collection(self):
        """Display the user's character collection."""
        print("🎴 Your Character Collection:")
        print("=" * 50)
        
        for char_name, count in self.user_collection.items():
            character = self.characters[char_name.lower().replace(" ", "_")]
            rarity_symbol = {
                "Common": "⚪",
                "Rare": "🔵", 
                "Epic": "🟣",
                "Legendary": "🟡",
                "Mythic": "🔴"
            }.get(character.rarity, "⚪")
            
            print(f"{rarity_symbol} {character.name} ({character.anime}) x{count}")
            print(f"   Strategy: {character.strategy_type}")
            print(f"   Power: {character.power_level} | Win Rate: {character.win_rate:.1%}")
            print(f"   Ability: {character.special_ability}")
            print()
    
    def generate_team_composition_report(self) -> Dict[str, Any]:
        """Generate a report on optimal team composition based on collected characters."""
        if not self.user_collection:
            return {"status": "No characters collected yet"}
        
        owned_characters = []
        for char_name in self.user_collection.keys():
            char_key = char_name.lower().replace(" ", "_")
            if char_key in self.characters:
                owned_characters.append(self.characters[char_key])
        
        # Sort by power level and win rate
        owned_characters.sort(key=lambda x: (x.power_level, x.win_rate), reverse=True)
        
        # Recommend top 3 for team
        recommended_team = owned_characters[:3]
        
        # Calculate team stats
        avg_power = np.mean([char.power_level for char in recommended_team])
        avg_win_rate = np.mean([char.win_rate for char in recommended_team])
        
        report = {
            "recommended_team": [
                {
                    "name": char.name,
                    "anime": char.anime,
                    "strategy_type": char.strategy_type,
                    "power_level": char.power_level,
                    "win_rate": char.win_rate,
                    "rarity": char.rarity
                } for char in recommended_team
            ],
            "team_stats": {
                "average_power": avg_power,
                "average_win_rate": avg_win_rate,
                "team_synergy": "Excellent" if avg_power > 80 else "Good" if avg_power > 60 else "Fair"
            },
            "collection_stats": {
                "total_characters": len(self.user_collection),
                "total_copies": sum(self.user_collection.values()),
                "rarity_distribution": self._get_rarity_distribution()
            }
        }
        
        return report
    
    def _get_rarity_distribution(self) -> Dict[str, int]:
        """Get the distribution of rarities in the user's collection."""
        rarity_count = {"Common": 0, "Rare": 0, "Epic": 0, "Legendary": 0, "Mythic": 0}
        
        for char_name in self.user_collection.keys():
            char_key = char_name.lower().replace(" ", "_")
            if char_key in self.characters:
                rarity = self.characters[char_key].rarity
                rarity_count[rarity] += 1
        
        return rarity_count
    
    def save_character_portraits(self, output_dir: str = "/home/ubuntu/fusion-project/python-backend/visualizations/"):
        """Save all character portraits as SVG files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for char_key, character in self.characters.items():
            filename = f"{output_dir}portrait_{char_key}.svg"
            with open(filename, 'w') as f:
                f.write(character.portrait_svg)
            print(f"Saved portrait: {filename}")
    
    def create_gacha_visualization(self):
        """Create a visualization of gacha rates and character rarities."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gacha rates pie chart
        rarities = list(self.gacha_config.rarity_rates.keys())
        rates = list(self.gacha_config.rarity_rates.values())
        colors = ['#808080', '#4169e1', '#9932cc', '#ffd700', '#ff1493']
        
        ax1.pie(rates, labels=rarities, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Gacha Pull Rates', fontsize=14, fontweight='bold')
        
        # Character power levels
        char_names = [char.name.split()[0] for char in self.characters.values()]
        power_levels = [char.power_level for char in self.characters.values()]
        rarity_colors = {
            'Common': '#808080',
            'Rare': '#4169e1', 
            'Epic': '#9932cc',
            'Legendary': '#ffd700',
            'Mythic': '#ff1493'
        }
        bar_colors = [rarity_colors[char.rarity] for char in self.characters.values()]
        
        bars = ax2.bar(char_names, power_levels, color=bar_colors)
        ax2.set_title('Character Power Levels', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Character')
        ax2.set_ylabel('Power Level')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, power in zip(bars, power_levels):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{power}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"/home/ubuntu/fusion-project/python-backend/visualizations/anime_gacha_system_{timestamp}.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Gacha visualization saved to: {filepath}")

# Example Usage
if __name__ == "__main__":
    # Initialize the anime lore system
    anime_system = AnimeLoreIntegration()
    
    print("🎌 Anime Lore Integration System Initialized!")
    print(f"💰 Starting Currency: {anime_system.user_currency}")
    
    # Save character portraits
    print("\n🎨 Generating character portraits...")
    anime_system.save_character_portraits()
    
    # Perform some gacha pulls
    print("\n🎰 Performing gacha pulls...")
    try:
        # Single pull
        pulled_chars = anime_system.perform_gacha_pull(1)
        print(f"Single pull result: {pulled_chars[0].name} ({pulled_chars[0].rarity})")
        
        # 10-pull
        pulled_chars = anime_system.perform_gacha_pull(10)
        print(f"\n10-pull results:")
        for char in pulled_chars:
            rarity_symbol = {
                "Common": "⚪",
                "Rare": "🔵", 
                "Epic": "🟣",
                "Legendary": "🟡",
                "Mythic": "🔴"
            }.get(char.rarity, "⚪")
            print(f"  {rarity_symbol} {char.name} ({char.rarity})")
        
        print(f"\n💰 Remaining Currency: {anime_system.user_currency}")
        
        # Display collection
        print("\n" + "="*50)
        anime_system.display_character_collection()
        
        # Generate team composition report
        print("📊 Team Composition Analysis:")
        report = anime_system.generate_team_composition_report()
        
        if "recommended_team" in report:
            print("\n🏆 Recommended Team:")
            for i, member in enumerate(report["recommended_team"], 1):
                print(f"{i}. {member['name']} - {member['strategy_type']}")
                print(f"   Power: {member['power_level']} | Win Rate: {member['win_rate']:.1%}")
            
            print(f"\n📈 Team Stats:")
            print(f"Average Power: {report['team_stats']['average_power']:.1f}")
            print(f"Average Win Rate: {report['team_stats']['average_win_rate']:.1%}")
            print(f"Team Synergy: {report['team_stats']['team_synergy']}")
        
        # Create visualization
        print("\n📊 Creating gacha visualization...")
        anime_system.create_gacha_visualization()
        
    except ValueError as e:
        print(f"Error: {e}")
    
    print("\n🎯 Anime lore integration complete! Your characters are ready for battle!")

