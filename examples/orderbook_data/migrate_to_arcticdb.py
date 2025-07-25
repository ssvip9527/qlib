# 版权所有 (c) 微软公司。
# 根据 MIT 许可证授权。
"""
从 Arctic 迁移到 ArcticDB 的工具脚本

此脚本帮助用户将 Arctic 数据库中的数据迁移到 ArcticDB。

用法:
    python migrate_to_arcticdb.py --mongo_uri="mongodb://localhost:27017" --arcticdb_uri="lmdb:///tmp/arcticdb" --libraries=ticks,transaction,order
"""

import argparse
import logging
import time
from typing import List, Optional

# 导入 Arctic 和 ArcticDB
try:
    from arctic import Arctic as OldArctic
    from arctic import CHUNK_STORE
except ImportError:
    raise ImportError("请先安装 arctic: pip install arctic")

try:
    import arcticdb as adb
except ImportError:
    raise ImportError("请先安装 arcticdb: pip install arcticdb")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("migration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("arctic_migration")


def migrate_library(old_arctic: OldArctic, new_arctic: adb.Arctic, library_name: str, batch_size: int = 100) -> None:
    """
    迁移单个库从 Arctic 到 ArcticDB
    
    参数:
        old_arctic: Arctic 实例
        new_arctic: ArcticDB 实例
        library_name: 要迁移的库名
        batch_size: 每批处理的符号数量
    """
    logger.info(f"开始迁移库: {library_name}")
    
    # 检查源库是否存在
    if library_name not in old_arctic.list_libraries():
        logger.error(f"源库 {library_name} 不存在")
        return
    
    # 获取源库
    old_lib = old_arctic[library_name]
    
    # 确保目标库存在
    if library_name not in new_arctic.list_libraries():
        logger.info(f"创建目标库: {library_name}")
        new_arctic.create_library(library_name)
    
    # 获取目标库
    new_lib = new_arctic[library_name]
    
    # 获取所有符号
    symbols = old_lib.list_symbols()
    total_symbols = len(symbols)
    logger.info(f"库 {library_name} 中共有 {total_symbols} 个符号需要迁移")
    
    # 批量处理符号
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    for i in range(0, total_symbols, batch_size):
        batch = symbols[i:i+batch_size]
        logger.info(f"处理批次 {i//batch_size + 1}/{(total_symbols + batch_size - 1)//batch_size}, 符号 {i+1}-{min(i+batch_size, total_symbols)}")
        
        for symbol in batch:
            try:
                # 检查符号是否已存在于目标库
                if symbol in new_lib.list_symbols():
                    logger.info(f"符号 {symbol} 已存在于目标库，跳过")
                    success_count += 1
                    continue
                
                # 读取数据
                data = old_lib.read(symbol)
                
                # 写入 ArcticDB
                new_lib.write(symbol, data)
                
                success_count += 1
                if success_count % 10 == 0:
                    logger.info(f"已成功迁移 {success_count} 个符号")
            
            except Exception as e:
                logger.error(f"迁移符号 {symbol} 失败: {str(e)}")
                error_count += 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"库 {library_name} 迁移完成:")
    logger.info(f"  - 总符号数: {total_symbols}")
    logger.info(f"  - 成功: {success_count}")
    logger.info(f"  - 失败: {error_count}")
    logger.info(f"  - 耗时: {duration:.2f} 秒")


def migrate_all(mongo_uri: str, arcticdb_uri: str, libraries: Optional[List[str]] = None, batch_size: int = 100) -> None:
    """
    迁移所有指定的库
    
    参数:
        mongo_uri: MongoDB URI
        arcticdb_uri: ArcticDB URI
        libraries: 要迁移的库列表，如果为None则迁移所有库
        batch_size: 每批处理的符号数量
    """
    logger.info(f"开始从 {mongo_uri} 迁移到 {arcticdb_uri}")
    
    # 连接源 Arctic
    old_arctic = OldArctic(mongo_uri)
    
    # 连接目标 ArcticDB
    new_arctic = adb.Arctic(arcticdb_uri)
    
    # 获取要迁移的库列表
    if libraries is None:
        libraries_to_migrate = old_arctic.list_libraries()
    else:
        libraries_to_migrate = libraries
        # 验证库是否存在
        for lib in libraries_to_migrate:
            if lib not in old_arctic.list_libraries():
                logger.warning(f"库 {lib} 在源 Arctic 中不存在，将被跳过")
    
    logger.info(f"将迁移以下库: {', '.join(libraries_to_migrate)}")
    
    # 迁移每个库
    for lib_name in libraries_to_migrate:
        migrate_library(old_arctic, new_arctic, lib_name, batch_size)
    
    logger.info("所有库迁移完成")


def main():
    parser = argparse.ArgumentParser(description="从 Arctic 迁移到 ArcticDB 的工具")
    parser.add_argument("--mongo_uri", type=str, default="mongodb://localhost:27017", help="MongoDB URI")
    parser.add_argument("--arcticdb_uri", type=str, default="lmdb:///tmp/arcticdb", help="ArcticDB URI")
    parser.add_argument("--libraries", type=str, help="要迁移的库，用逗号分隔。如果不指定，则迁移所有库")
    parser.add_argument("--batch_size", type=int, default=100, help="每批处理的符号数量")
    
    args = parser.parse_args()
    
    libraries = args.libraries.split(",") if args.libraries else None
    
    migrate_all(args.mongo_uri, args.arcticdb_uri, libraries, args.batch_size)


if __name__ == "__main__":
    main()