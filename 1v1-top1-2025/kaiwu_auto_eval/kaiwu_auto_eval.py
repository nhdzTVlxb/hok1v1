# kaiwu_auto_eval.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import re
import time
import asyncio
import pandas
import config as cfg
from typing import Dict, Optional, Set
from calc_str_timedelta import add_two_str_time_to_str

from playwright.async_api import async_playwright, Browser, Page
PATH_LOGS_DIR = Path(__file__).parent / "logs"
PATH_LOGS_DIR.mkdir(parents=True, exist_ok=True)

class WinrateManager:
    def __init__(self):
        self.added_models = set()
        self.path_csv = PATH_LOGS_DIR / f"winrates_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        self.df = pandas.DataFrame(columns=['模型版本', '训练步数', '当前训练时长', '总训练时长', '开始时间'])
    
    def update_winrate(self, our_model: str, opponent_model: str, win_num: int, total_num: int, train_time: str, start_time: str):
        if (our_model, opponent_model, start_time) in self.added_models:
            return
        self.added_models.add((our_model, opponent_model, start_time))
        our_model_version, our_model_steps = our_model.rsplit('_', 1)
        total_train_time = add_two_str_time_to_str(train_time, cfg.previous_train_time)
        print(f"新的评估信息: 模型 '{our_model}' vs '{opponent_model}' 胜场/总场次: {win_num}/{total_num} 胜率: {win_num/total_num:.2%} 当前训练时长: {train_time} 总训练时长: {total_train_time} 开始时间: {start_time}")

        # 判断当前模型信息是否存在, 不存在则创建
        model_existed_fn = lambda: (self.df['模型版本'] == our_model_version) & (self.df['训练步数'] == our_model_steps)
        if not model_existed_fn().any():
            self.df = pandas.concat([self.df, pandas.DataFrame({'模型版本': [our_model_version], '训练步数': [our_model_steps], '当前训练时长': [train_time], '总训练时长': [total_train_time], '开始时间': [start_time]})], ignore_index=True)

        self.df.loc[model_existed_fn(), opponent_model] = f"={win_num}/{total_num}"
        self.df.loc[model_existed_fn(), f"{opponent_model}_胜率"] = f"{win_num/total_num:.2%}"
        self.df = self.df.sort_values(by=['模型版本', '训练步数']).reset_index(drop=True)
        self.df.to_csv(self.path_csv, index=False, encoding='utf-8')

class KaiwuAutoEval:
    """
    一个使用 Playwright 自动执行腾讯开悟平台模型评估流程的类。

    这个类实现了登录、导航、查找最新模型、提交模型、
    创建评估任务等一系列自动化操作。
    """

    def __init__(self, browser: Browser):
        """
        初始化 KaiwuAutoEval 实例。

        Args:
            browser (Browser): 一个 Playwright Browser 实例。
        """
        self.browser = browser
        self.page: Optional[Page] = None
        self.base_url = cfg.base_url
        
        # 用于存储已添加或已评估过的模型，避免重复操作
        self.add_model_set: Set[str] = cfg.add_model_set
        
        # 存储待评估的对手模型及其在任务名称中的缩写
        self.eval_model_dict: Dict[str, str] = cfg.eval_model_dict

        # 存储模型胜率信息
        self.winrate_manager = WinrateManager()

    async def login(self):
        """
        第一步：导航到开悟平台并等待用户手动登录。

        由于登录过程可能涉及验证码或第三方认证，此函数将打开登录页面，
        并等待用户完成登录操作。脚本将在登录成功并导航到主仪表板后继续。

        Args:
            None
        """
        print("步骤 1: 正在导航到登录页面...")
        self.page = await self.browser.new_page()
        await self.page.goto(self.base_url)
        print(f"页面 {await self.page.title()} 已加载")

        try:
            email_input = self.page.get_by_placeholder(re.compile("邮箱"))
            await email_input.fill(cfg.user_email)
            password_input = self.page.get_by_placeholder(re.compile("密码"))
            await password_input.fill(cfg.user_password)
            print(f"已输入邮箱: {cfg.user_email}, 密码: {cfg.user_password}")
            agreement_checkbox = self.page.get_by_role("checkbox")
            await agreement_checkbox.check()
            login_button = self.page.get_by_role("button", name=re.compile(r"登\s*录"))
            await login_button.click()

        except Exception as e:
            print(f"自动登陆失败: {e}")
            raise
        
        # 等待用户登录成功后跳转到特定页面，这里以某个仪表盘元素的出现为标志
        # 您需要根据实际页面情况修改下面的选择器
        await self.page.wait_for_selector(f"text={cfg.eval_page_title}", timeout=300000) # 等待5分钟
        print(f"打开界面 {await self.page.title()}，登录成功，继续执行...")

    async def navigate_to_model_list(self) -> bool:
        """
        第二步：从主页面导航到训练中的模型列表。

        此函数会点击进入训练界面，并打开第一个正在训练的模型的“模型列表”。

        Args:
            None

        Returns:
            bool: 如果成功导航到模型列表页面，则返回 True，否则返回 False。
        """
        print("步骤 2: 正在导航到模型列表...")
        await self.page.reload()
        # await self.page.wait_for_selector("text=集群训练", timeout=30000)
        # await self.page.click("text=集群训练")
        # await self.page.get_by_role("menuitem", name="集群训练").click()
        await self.page.goto(f"{self.base_url}/training")
        await self.page.locator("text=模型列表").first.click()
        print("已成功导航到模型列表。")
        return True

    async def find_and_process_newest_model(self) -> Optional[Dict[str, any]]:
        """
        第三步：查找并处理最新的、未被添加过的模型。

        此函数会读取模型列表，按“训练步数”降序排序，并找到第一个
        尚未存在于 `self.add_model_set` 中的模型。如果找不到新模型，
        它将等待一分钟后刷新页面重试。

        Args:
            None

        Returns:
            Optional[Dict[str, any]]: 如果找到新的模型，则返回该模型的信息字典；
                           如果找不到，则返回 None。
        """
        print("步骤 3: 正在查找最新的模型...")
        model_list = []
        # 等待抽屉和表格内容加载完成
        # .ant-drawer-body 定位到抽屉的身体部分，确保内容可见
        drawer_body = self.page.locator(".ant-drawer-body")
        await drawer_body.wait_for(timeout=10000)  # 等待10秒

        # 在抽屉内部找到 <h5> 标签
        title_locator = drawer_body.locator("h5.ant-typography")
        model_version_title = await title_locator.inner_text()
        print(f"成功获取到模型列表标题: {model_version_title}")

        # 等待 tbody 中第一条 class='ant-table-row' 的 tr 元素出现
        try:
            first_row_locator = drawer_body.locator(".ant-table-tbody tr.ant-table-row").first
            await first_row_locator.wait_for(timeout=30000) # 等待30秒
        except Exception as e:
            raise Exception("当前模型列表没有模型, 重新刷新...")
        print("模型列表数据已加载。")

        # 获取表格中的所有行，忽略第一行的测量行
        rows = drawer_body.locator(".ant-table-tbody tr.ant-table-row")
        
        row_count = await rows.count()
        if row_count == 0:
            print("表格中没有找到模型数据。")
            return None

        print(f"在当前页面找到 {row_count} 个模型，开始解析...")

        new_model_info = None

        # 遍历每一行来提取数据
        for i in range(row_count):
            row = rows.nth(i)
            # 获取当前行的所有单元格 (td)
            cells = row.locator("td")

            # 按顺序提取每个单元格的文本
            duration = await cells.nth(0).inner_text()
            steps_text = await cells.nth(1).inner_text()
            size = await cells.nth(2).inner_text()

            # 在第四个单元格内查找操作按钮
            action_cell = cells.nth(3)
            download_button = action_cell.get_by_role('button', name='下载')
            submit_button = action_cell.get_by_role('button', name='提交到模型管理')

            # 将提取的数据存入字典
            model_info = {
                'version': model_version_title,
                'name': f'{model_version_title}_{steps_text}',
                'duration': duration,
                'steps': int(steps_text), # 将步数转为整数，方便后续排序
                'size': size,
                'download_button': download_button,
                'submit_button': submit_button,
                'row_locator': row # 保存整行的定位器，方便后续操作
            }
            model_list.append(model_info)

            if new_model_info is None or model_info['steps'] > new_model_info['steps']:
                if model_info['name'] not in self.add_model_set:
                    self.add_model_set.add(model_info['name'])
                    new_model_info = model_info

        print(f"成功解析了 {len(model_list)} 个模型，当前最新模型步数为 {new_model_info['steps'] if new_model_info else '无'}")
        # pprint(model_list)
        # print(f"最新模型为 {new_model_info}")

        return new_model_info

    async def submit_model_to_management(self, model_info: Dict[str, any]):
        """
        第四步：将新发现的模型提交到模型管理。

        函数会点击“提交到模型管理”，并使用推荐的命名格式（版本号_步数）
        来命名模型。

        Args:
            model_info (Dict[str, any]): 从上一步获取的最新模型的信息字典。
        """
        model_name = f"{model_info['version']}_{model_info['steps']}"
        print(f"步骤 4: 正在提交模型 '{model_name}' 到模型管理...")
        await model_info['submit_button'].click()
        await self.page.get_by_placeholder('输入模型名称').fill(model_name)
        await self.page.get_by_role('button', name=re.compile(r"^提\s*交$")).click()

        print("模型提交成功，等待3秒自动跳转到模型管理界面...", end="", flush=True)
        await self.page.wait_for_url("**/model-manage", timeout=30000)
        print("已成功跳转到模型管理界面。")

    async def update_winrate(self):
        """
        在模型评估刷新过程中更新模型胜率信息。
        """
        print("开始更新胜率")
        battle_tasks = self.page.locator("[class*='battleTaskItem']")
        for i in range(await battle_tasks.count()):
            task = battle_tasks.nth(i)
            finish_tag = task.get_by_text("已完成")
            if not await finish_tag.is_visible():
                continue
            submit_time = await task.locator("div:has-text('提交于') + div").first.inner_text()
            camps = task.locator("[class*='camp___']")
            model_infos = []
            for j in range(await camps.count()):
                camp = camps.nth(j)
                model_name = await camp.locator("span:has-text('阵营模型') + span").inner_text()
                model_train_time = await camp.locator("[class*='campInfo___']").locator("span").first.inner_text()
                model_infos.append((model_name, model_train_time))
            score_box = task.locator("[class*='scoreBox___']")
            async def get_score_by_label(label: str):
                value_locator = score_box.locator("[class*='main___']").filter(has_text=label).locator("[class*='value___']")
                return await value_locator.inner_text()
            a_wins_count = int(await get_score_by_label("A胜"))
            timeouts_count = int(await get_score_by_label("超时"))
            total_game_count = int(await get_score_by_label("总局数"))
            self.winrate_manager.update_winrate(
                our_model=model_infos[0][0],
                opponent_model=model_infos[1][0],
                win_num=a_wins_count,
                total_num=total_game_count - timeouts_count,
                train_time=model_infos[0][1],
                start_time=submit_time
            )

    async def wait_for_evaluation_finish(self):
        """
        等待当前评估任务完成。
        """
        await self.page.click("text=模型评估")
        first_task_item = self.page.locator('[class*="battleTaskItem"]').first
        await first_task_item.wait_for(timeout=30000)

        # 等待列表刷新完成
        list_locator = self.page.locator(".ant-pro-search-content")
        await list_locator.wait_for(timeout=10000)
        # 1. 创建一个定位器，用于查找“进行中”
        # locator_in_progress = list_locator.get_by_text("进行中")
        # 2. 创建另一个定位器，用于查找“待分配” (因为并行推理可能大于当前对局数, 因此通过待分配来判断是否全部启动)
        locator_pending = list_locator.get_by_text("等待中")
        # 3. 使用 .or_() 将它们合并。新的 status_locator 现在可以匹配任何一个
        # status_locator = locator_in_progress.or_(locator_pending)
        status_locator = locator_pending
        refresh_button = self.page.locator(".ant-btn-refresh")

        await self.update_winrate()

        while await status_locator.count() > 0:
            print(f"检测到有等待中的{await status_locator.count()}个评估任务，等待其完成...{time.strftime('%Y%m%d-%H%M%S')}")
            await refresh_button.click()
            await first_task_item.wait_for(timeout=30000)
            await asyncio.sleep(10)  # 等待10秒后再检查
        
        await self.update_winrate()

    async def navigate_and_create_evaluation(self, model_name: str):
        """
        第五步：导航到模型评估界面并准备创建新任务。

        此函数会跳转到“模型评估”页面，并检查当前是否有正在进行的任务。
        如果有，它会等待任务结束后再继续。

        Args:
            model_name (str): 最新模型的名称，用于后续命名。
        """
        print("步骤 5: 正在导航到模型评估界面...")
        await self.wait_for_evaluation_finish()
        
        # 一次性将当前所有对手模型加入评估任务
        for opponent_name, opponent_abbr in self.eval_model_dict.items():
            result = await self.configure_evaluation_task(model_name, opponent_name, opponent_abbr)
            while not result:
                print(f"任务 {model_name} vs {opponent_name} 创建失败，重试...")
                result = await self.configure_evaluation_task(model_name, opponent_name, opponent_abbr)

        await self.wait_for_evaluation_finish()

    async def _select_camp_details(self, camp_label: str, model_to_search: str):
        """
        一个修正后的辅助函数，用于为特定阵营选择模型和全部三个英雄。
        此版本能正确处理 Ant Design 的动态搜索下拉框。

        Args:
            camp_label (str): 阵营的标签, 例如 "阵营A 模型".
            model_to_search (str): 需要搜索的模型名称.
        """
        print(f"  - 正在配置 {camp_label.split(' ')[0]}:")
        
        # 定位到特定阵营的区域
        print(f"    - 正在定位 '{camp_label}' 区域...")
        camp_section = self.page.locator(f".ant-form-item:has-text('{camp_label}')")

        # ------------------- 核心交互部分：选择模型 -------------------
        print(f"    - 正在搜索模型: '{model_to_search}'...")

        # 存在两个选择，“选择模型”和“选择阵容”
        selectors = camp_section.locator(".ant-select-selector")
        
        # 步骤 1: 点击“选择模型”的占位符，触发下拉菜单的出现
        await selectors.nth(0).click()

        # 步骤 2: 定位并填充真实的搜索框
        # 注意：我们使用 get_by_placeholder 来定位新出现的搜索框，这是一个非常稳定的方法
        await camp_section.get_by_placeholder("搜索模型名称").fill(model_to_search)

        # 步骤 3: 定位并点击第一个结果项
        # 我们使用 '.ant-select-item-option' 这个稳定的类名来定位所有选项
        # 然后用 .first 选取第一个，并点击它。
        # Playwright 的 .click() 会自动等待该元素出现，确保了稳定性。
        print("    - 等待并选择搜索结果...")
        await camp_section.locator(".ant-select-item-option").first.click()
        print("    - 模型已选择。")
        # ------------------- 交互结束 -------------------

        # 3. 选择阵容（这部分逻辑保持不变）
        print("    - 正在选择阵容...")
        await selectors.nth(1).click()
        
        # 逐个点击英雄选项
        await camp_section.get_by_text("1.后羿").click()
        await camp_section.get_by_text("2.李元芳").click()
        await camp_section.get_by_text("3.虞姬").click()
        
        await self.page.keyboard.press("Escape")
        print("    - 阵容已选择。")

    async def configure_evaluation_task(self, model_name: str, opponent_name: str, opponent_abbr: str):
        """
        第六步：配置并启动新的评估任务。

        此函数会设置评估任务的各个参数，包括任务名称、对战双方的模型、
        英雄阵容和评估局数。

        Args:
            model_steps (str): 最新模型的训练步数。
            opponent_steps (str): 对手模型的训练步数。
            opponent_abbr (str): 对手模型的简称。

        """
        task_name = f"{model_name} {opponent_abbr}" # e.g., "v1_2_2_191475 b1"
        print(f"步骤 6: 正在配置评估任务 '{task_name}'...")
        print("  - 刷新页面以确保最新状态...")
        await self.page.reload()
        await self.page.wait_for_selector("text=新增评估任务", timeout=30000)

        print("  - 开始创建新评估任务...")
        # A. Click the "Add Task" button to open the sidebar
        await self.page.get_by_role("button", name="新增评估任务").click()

        # B. Wait for the sidebar (drawer) to be visible before proceeding
        drawer = self.page.locator(".ant-drawer-body")
        await drawer.wait_for(timeout=10000)
        
        # C. Fill in the task name
        await drawer.get_by_placeholder("输入任务名称").fill(task_name)

        # D. Use the helper function to configure Camp A
        await self._select_camp_details(camp_label="阵营A 模型", model_to_search=model_name)

        # E. Use the helper function to configure Camp B
        await self._select_camp_details(camp_label="阵营B 模型", model_to_search=opponent_name)

        # F. Set the number of evaluation games
        print("  - Setting evaluation rounds to 5...")
        # The number input has a role of "spinbutton"
        await drawer.get_by_role("spinbutton").fill(f"{cfg.battle_rounds}")

        # G. Click the final submit button
        print("  - Submitting new evaluation task...")
        await drawer.get_by_role("button", name="完成新增").click()

        # H. Wait for the drawer to disappear, confirming submission
        await drawer.wait_for(state="hidden", timeout=5000)

        max_retries = 3
        while await drawer.is_visible() and max_retries > 0:
            print(f"  - Task '{task_name}' drawer 仍然可见, 重新尝试点击提交按钮...")
            await drawer.get_by_role("button", name="完成新增").click()
            await drawer.wait_for(state="hidden", timeout=5000)
            max_retries -= 1
        
        if await drawer.is_visible():
            print(f"  - Task '{task_name}' drawer 仍然可见, 提交失败.")
            return False

        print(f"  - Task '{task_name}' successfully created.")
        return True

    async def run(self):
        """
        运行完整自动化流程的主函数。

        此函数会按顺序调用所有步骤，并包含主循环逻辑。
        """
        await self.login()

        while True:
            try:
                if not await self.navigate_to_model_list():
                    raise Exception("导航到模型列表失败，请检查页面结构。")

                model_info = await self.find_and_process_newest_model()
                if model_info:
                    await self.submit_model_to_management(model_info)
                    await self.navigate_and_create_evaluation(model_info['name'])
                    print("完成一轮评估，流程将从头开始...")
                else:
                    print("本轮未发现新模型，继续监控...")
                # 等待片刻，避免过于频繁的请求
                await asyncio.sleep(5)

            except Exception as e:
                print(f"在执行过程中发生错误: {e}")
                print("将在 30 秒后重试...")
                await asyncio.sleep(30)

async def main():
    """
    主入口函数，用于启动 Playwright 并运行自动化任务。
    """
    async with async_playwright() as p:
        # 启动一个非无头模式的浏览器，方便观察和调试
        # browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        # 正常使用直接用无头模式即可
        browser = await p.chromium.launch(headless=True)
        
        auto_evaluator = KaiwuAutoEval(browser)
        
        await auto_evaluator.run()
        
        # 在实际使用中，您可能希望脚本一直运行，所以下面这行可以注释掉
        # await browser.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已被用户中断。")
